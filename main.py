import os

import gensim.downloader
import torch.cuda
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors, Word2Vec
from gensim.summarization.summarizer import summarize
from rouge_score.rouge_scorer import RougeScorer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, RegexpTokenizer
import matplotlib.pyplot as plt
from seq2seq import Seq2Seq, CustomDataset
from collections import Counter
from typing import List
from transformers import pipeline, AutoTokenizer, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, Seq2SeqTrainer, \
    Seq2SeqTrainingArguments, SummarizationPipeline
import evaluate

MODEL_NAME = "pretrained.model"
TRANSFORMER_MODEL_DIR = "transformer_model"
tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL_DIR)
rouge = evaluate.load("rouge")

def preprocess_data(data: pd.Series) -> List[List[str]]:
    """
    Transforms a Series of data into a list of list of word vectors for each document

    :param data:
    :return: list of list of vectors
    """
    data = data.str.lower()
    clean_texts = []
    for i, text in data.items():
        remove_punct = RegexpTokenizer(r'\w+')
        clean_texts.append(remove_punct.tokenize(text))

    return clean_texts


def rogue_score(predictions: List[List[str]], references: List[List[str]], scorer: RougeScorer):
    """
    Returns a DataFrame of scores for each prediction/reference pair

    :param predictions:
    :param references:
    :param scorer:
    :return: A DataFrame containing the prediction, reference and scores for each example
    """

    scores = pd.DataFrame(columns=['prediction', 'reference', 'precision', 'recall', 'fmeasure'])
    for predicted, reference in zip(predictions, references):
        predicted = ' '.join(predicted)
        reference = ' '.join(reference)

        score = scorer.score(predicted, reference)['rouge2']
        score_dict = pd.DataFrame([{'prediction': predicted, 'reference': reference, 'precision': score.precision,
                                    'recall': score.recall, 'fmeasure': score.fmeasure}])

        scores = pd.concat([scores, score_dict], ignore_index=True)

    return scores


def bleu_score(predictions: List[List[str]], references: List[List[str]]):

    scores = pd.DataFrame(columns=['prediction', 'reference', 'bleu_score'])
    smooth = SmoothingFunction()
    for prediction, reference in zip(predictions, references):
        score = sentence_bleu(reference, prediction, smoothing_function=smooth.method3)
        score_dict = pd.DataFrame([{"prediction": ' '.join(prediction), "reference": ' '.join(reference), "bleu_score": score}])
        scores = pd.concat([scores, score_dict], ignore_index=True)

    return scores


def perform_baseline(test_data: pd.DataFrame, scorer: RougeScorer):
    summaries = []
    references = []
    for i, row in test_data.iterrows():
        summaries.append(summarize(row['article'].replace('.', '\n').replace('?', '\n').replace('!', '\n')))
        references.append(row['highlights'])

    print(summaries)

    rouge_scores = rogue_score(summaries, references, scorer)
    bleu_scores = bleu_score(summaries, references)

    scores = pd.merge(rouge_scores, bleu_scores, on=['prediction', 'reference'])

    return scores


def plot_scores(scores):
    plt.figure(1)
    plt.plot(range(len(scores)), scores['bleu_score'], 'b')
    plt.title("Bleu Scores")
    plt.ylabel("Score")
    plt.xlabel("Example #")
    plt.show()

    plt.figure(2)
    plt.plot(range(len(scores)), scores['fmeasure'], 'g', label='F1 Score')
    # plt.plot(range(len(scores)), scores['recall'], 'r--', label='Recall')
    # plt.plot(range(len(scores)), scores['precision'], 'y--', label='Precision')
    plt.title("ROUGE-2 Scores")
    plt.ylabel("Score")
    plt.xlabel("Example #")
    plt.legend()
    plt.show()


def print_results(scores, model_name):
    print(f"{model_name} Mean F1:  {scores['fmeasure'].mean()}")
    print(f"{model_name} Mean Bleu Score: {scores['bleu_score'].mean()}")
    print(f"{model_name} Mean Precision: {scores['precision'].mean()}")
    print(f"{model_name} Mean Recall: {scores['recall'].mean()}\n")


def preprocess_for_transformer(data: pd.DataFrame, tokenizer: AutoTokenizer):
    inputs = ["summarize: " + row for row in data["article"]]
    highlights = [row for row in data["highlights"]]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)

    references = tokenizer(highlights, max_length=128, truncation=True)
    #model_inputs["labels"] = references["input_ids"]

    return CustomDataset(model_inputs, references["input_ids"])


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}


def train_and_eval_transformer(train_data: pd.DataFrame, eval_data: pd.DataFrame, test_data:pd.DataFrame, scorer, train: bool = False) -> pd.DataFrame:
    train_dataset = preprocess_for_transformer(train_data, tokenizer)
    eval_dataset = preprocess_for_transformer(eval_data, tokenizer)

    data_batcher = DataCollatorForSeq2Seq(tokenizer=tokenizer, model="t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained(TRANSFORMER_MODEL_DIR)

    if train:
        training_args = Seq2SeqTrainingArguments(
            output_dir="./results",
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            weight_decay=0.01,
            save_total_limit=3,
            num_train_epochs=4,
            predict_with_generate=True,
            # fp16=True,
            # push_to_hub=True,
        )

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_batcher,
            compute_metrics=compute_metrics,
        )

        trainer.train()
        trainer.save_model("transformer_model")
        trainer.save_metrics("transformer_metrics", "all", ["rouge1", "rouge2"])

    summarizer = SummarizationPipeline(model=model, tokenizer=tokenizer)

    summaries = []
    references = []
    for i, row in test_data.iterrows():
        summaries.append(summarizer(row["article"], min_length=50, max_length=128)[0]["summary_text"])
        references.append(row["highlights"])

    summaries = [sum.split(' ') for sum in summaries]
    references = [ref.split(' ') for ref in references]

    print(list(zip(summaries[:10], references[:10])))

    rouge_scores = rogue_score(summaries, references, scorer)
    bleu_scores = bleu_score(summaries, references)

    scores = pd.merge(rouge_scores, bleu_scores, on='reference')

    return scores


if __name__ == '__main__':
    print("Beginning Encoder-Decoder Model Training")

    train = pd.read_csv("data/train.csv")
    validation = pd.read_csv("data/validation.csv")
    test = pd.read_csv("data/test.csv")

    training_inputs = preprocess_data(train["article"][:3000])
    training_references = preprocess_data(train["highlights"][:3000])
    validation_inputs = preprocess_data(validation["article"][:3000])
    validation_references = preprocess_data(validation["highlights"][:3000])

    print("Read in inputs... Beginning training Transformer")

    scorer = RougeScorer(["rouge2"], use_stemmer=True)

    transformer_test_scores = train_and_eval_transformer(train[:100], validation[:100], train[:500], scorer, False)
    print_results(transformer_test_scores, "Transformer")
    transformer_test_scores.to_csv("transformer_test_scores.csv")

    print("Finished Training Transformer...")

    vector_model = Word2Vec(sentences=training_inputs, size=300, window=5, min_count=1, workers=4, sg=1)
    vector_model.save("original_w2v.npy")
    # vector_model = Word2Vec.load("original_w2v.npy")

    print("Trained Word2Vec Model... Beginning Training Custom Seq2Seq Model")
    # baseline_scores = perform_baseline(test, scorer)
    # print_results(baseline_scores, "Baseline Model")

    seq2seq_model = Seq2Seq(300, 100, vector_model, .01)

    losses = seq2seq_model.train_model(training_inputs, training_references, epochs=10)
    seq2seq_model.save()

    print("Done Training model, Starting Evaluation:")

    results = seq2seq_model.predict(validation_references)
    rouge_scores = rogue_score(results, validation_references, scorer)
    bleu_scores = bleu_score(results, validation_references)

    seq2seq_scores = pd.merge(rouge_scores, bleu_scores, on='reference')
    print_results(seq2seq_scores, "Seq2Seq Model")
    seq2seq_scores.to_csv("validation_scores.csv")

    plt.plot(range(len(losses)), losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Encoder-Decoder Training Loss")
    plt.show()

    plt.savefig("Loss Chart")
