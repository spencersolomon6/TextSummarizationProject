# Comparison of Text Summarization Methods

For our final project for CS6120: NLP, we decided to compare several ways to summarize text.

We have the following models:
- Sequence-to-Sequence LSTM (abstractive)
- LDA (extractive)
- LexRank (extractive)

The baseline for comparison was the implementation of the TextRank algorithm in the `gensim` package.

## Dataset
Please download the dataset from the following link:
https://www.kaggle.com/datasets/gowrishankarp/newspaper-text-summarization-cnn-dailymail

## Installation Instructions
You'll need to use __Python 3.9__ for this project.
Then run the following command to install the necessary packages:

```sh
pip install -r requirements.txt
```

After that, you can run any of the following 3 files to see the evaluation of a certain model:
- `lda_eval.py`
- `lex_rank_eval.py`
- `lstm_eval.py`

For each evaluation script, there is a comment at the top of the file which describes how to prepare the data so the file can read it. The `lda_eval.py` and `lstm_eval.py` scripts expect the data in the same format, but `lex_rank_eval.py` expects it differently by expecting a zipped archive from which it reads the data.

## Authors
- Spencer Solomon
- Seamus Rioux
- Talal Siddiqui