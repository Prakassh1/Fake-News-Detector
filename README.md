# Paper Publication

https://ieeexplore.ieee.org/document/9743010

# Fake-News-Detector
 A basic machine learning based News detector to identify the authencity of the news articles.
 Algorithms like Pasiive-Aggressive, Naive Bayes, Support Vector Machine and Random Forest were used

# Overview
The topic of fake news detection on social media has recently attracted tremendous attention. The basic countermeasure of comparing websites against a list of labeled fake news sources is inflexible, and so a machine learning approach is desirable. Our project aims to use Natural Language Processing to detect fake news directly, based on the text content of news articles.

# Dataset

    train.csv: A full training dataset with the following attributes:
        id: unique id for a news article
        title: the title of a news article
        author: author of the news article
        text: the text of the article; could be incomplete
        label: a label that marks the article as potentially unreliable
            1: unreliable
            0: reliable

    test.csv: A testing training dataset with all the same attributes at train.csv without the label.

# Steps



    Clone the repo to your local machine-
    > git clone git://github.com/Prakassh1/Fake-News-Detector.git
    > cd Fake-News-Detector

    Make sure you have all the dependencies installed-

    python 3.6+
    numpy
    tensorflow
    gensim
    pandas
    nltk
        For nltk, we recommend typing python.exe in your command line which will take you to the Python interpretor.
            Then, enter-
                >>> import nltk
                >>> nltk.download()}

    You're good to go now-
    > python svm.py


# Dataset Repo
  1.Datasets from Kaggle
