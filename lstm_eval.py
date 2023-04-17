import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from rouge_score.rouge_scorer import RougeScorer
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, Seq2SeqTrainer, \
    Seq2SeqTrainingArguments, SummarizationPipeline
import evaluate
from models.seq2seq import Seq2Seq
from utils import preprocess_data, preprocess_for_transformer, print_results, bleu_score, rogue_score, perform_baseline


MODEL_NAME = "pretrained.model"
TRANSFORMER_MODEL_DIR = "transformer_model"
tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL_DIR)
rouge = evaluate.load("rouge")


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

    summarizer = SummarizationPipeline(model=model, tokenizer=tokenizer)

    summaries = []
    references = []
    for i, row in test_data.iterrows():
        if i == 0:
            print(summarizer(row["article"], min_length=50, max_length=128)[0]["summary_text"])

        summaries.append(summarizer(row["article"], min_length=50, max_length=128)[0]["summary_text"])
        references.append(row["highlights"])

    summaries = [sum.split(' ') for sum in summaries]
    references = [ref.split(' ') for ref in references]

    for sum, ref in zip(summaries[:5], references[:5]):
        print(f"Summary: {sum} /// Reference: {ref}\n")

    rouge_scores = rogue_score(summaries, references, scorer)
    bleu_scores = bleu_score(summaries, references)

    scores = pd.merge(rouge_scores, bleu_scores, on='reference')

    return scores


if __name__ == '__main__':
    print("Starting reading in data...")

    train = pd.read_csv("data/train.csv")
    validation = pd.read_csv("data/validation.csv")
    test = pd.read_csv("data/test.csv")

    training_inputs = preprocess_data(train["article"][:1000])
    training_references = preprocess_data(train["highlights"][:1000])
    validation_inputs = preprocess_data(validation["article"][:1000])
    validation_references = preprocess_data(validation["highlights"][:1000])

    print("Read in inputs... Beginning performing Baseline")

    scorer = RougeScorer(["rouge1", "rouge2"], use_stemmer=True)
    baseline_scores = perform_baseline(test[:1000], scorer)
    print_results(baseline_scores, "Baseline")

    print("Training/Evaluating Transformer")

    transformer_test_scores = train_and_eval_transformer(train[:250], validation[:250], test[:1000], scorer, False)
    print_results(transformer_test_scores, "Transformer")
    transformer_test_scores.to_csv("transformer_test_scores.csv")

    print("Finished Training Transformer...")

    vector_model = Word2Vec(sentences=training_inputs, size=300, window=5, min_count=1, workers=4, sg=1)
    vector_model.save("original_w2v.npy")

    print("Trained Word2Vec Model... Beginning Training Custom Seq2Seq Model")

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
