# Text Sentiment Analysis: A Comparative Approach
**Note: only selected files from the studies mentioned below are included in this repository.**

This project was completed as part of a course at Ankara University. We chose the topic of binary Text Sentiment Analysis (Positive-Negative), a popular subject in Natural Language Processing (NLP).

## Overview ##
In this project, we explored various feature extraction methods and classification techniques for sentiment analysis:

- Feature Extraction Methods: BERT, Bag of Words (BoW), TF-IDF
- Classification Methods: SVM, Naive Bayes, XGBoost, Softmax, Logistic Regression
We evaluated our models using four different metrics: Accuracy, F1 Score, Precision, and Recall.

We used 2 different datasets which are IMDB Review Dataset and Sentiment140 dataset with 1.6 million tweets.

We first preprocessed the dataset in a way that suits us (Removing some special characters and phrases etc.)

After all experiments, `BERTForSequenceClassification()` class from `Transformers` library gave us the best scores (%90 accuracy). It has it's default classifier which is softmax. Although we tried to get BERT's last output layers and gave it to our ANN models which also included softmax, we got worse scores than `BERTForSequenceClassification()`'s built-in classifier.




