# Text Sentiment Anaysis: A comparative approach
**Since this is a hobby project, it was not written with a clean code approach. Moreover, of the studies mentioned below, only some files from the works have been added to the repo.**

This project was for the one of the courses of Ankara University. The project topic was up to us and we selected one of the popular topics in NLP(Natural Language Processing) which is binary Text Sentiment Analysis (Positive-Negative).

We tried to use some different feature extraction methods like BERT, BoW, TF-IDF and also classification methods such as SVM, NaiveBayes, XGBoost, Softmax, Logistic Regression etc. 

We used 4 different metrics which are Accuracy, F1 Score, Precision-Recall.

We used 2 different datasets which are IMDB Review Dataset and Sentiment140 dataset with 1.6 million tweets.

We first preprocessed the dataset in a way that suits us (Removing some special characters and phrases etc.)

After all experiments, `BERTForSequenceClassification()` class from `Transformers` library gave us the best scores (%90 accuracy). It has it's default classifier which is softmax. Although we tried to get BERT's last output layers and gave it to our ANN models which also included softmax, we got worse scores than `BERTForSequenceClassification()`'s built-in classifier.




