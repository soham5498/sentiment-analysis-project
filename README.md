# Sentiment Analysis using BERT

## Overview
This project implements sentiment analysis using a pre-trained BERT model for sequence classification. The dataset used is the IMDb movie reviews dataset, and the model is fine-tuned for binary sentiment classification (positive or negative reviews).

## Features
- Utilizes `transformers` library for a pre-trained BERT model.
- Loads and processes the IMDb dataset using the `datasets` library.
- Fine-tunes BERT for sentiment classification.
- Evaluates model performance using accuracy and precision-recall metrics.
- Uses GPU acceleration if available.

## Installation
Ensure you have Python 3.7+ installed. Then, install the required dependencies using:

```sh
pip install transformers datasets torch scikit-learn
```

## Usage
Run the notebook step by step to:
1. Load the IMDb dataset.
2. Tokenize the text using BERT tokenizer.
3. Fine-tune BERT for classification.
4. Evaluate the model's performance.

To execute the notebook, use Jupyter:

```sh
jupyter notebook Sentiment_analysis.ipynb
```

## Model Training
The training process involves:
- Initializing `BertForSequenceClassification`.
- Using `Trainer` API for fine-tuning.
- Applying `AdamW` optimizer and a learning rate scheduler.
- Evaluating the model after training.

## Evaluation
The model's performance is assessed using:
- Accuracy score
- Precision, Recall, and F1-score

Results are printed at the end of the training process.

## License
This project is open-source and can be used freely.



