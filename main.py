import os

import pandas as pd

import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

import joblib
import pickle

from sklearn.metrics import confusion_matrix, accuracy_score

path_to_tsv = "data/in/"
path_to_bow = "data/out/bow/"
path_to_classifier = "data/out/classifier/"
tsv_name = "a1_RestaurantReviews_HistoricDump.tsv"

def load_data(file):
    return pd.read_csv(file, delimiter='\t', quoting=3)

def preprocess_data(data):
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')

    corpus = []

    for i in range(len(data)):
        review = re.sub('[^a-zA-Z]', ' ', data['Review'][i])
        review = review.lower()
        review = review.split()
        review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
        review = ' '.join(review)
        corpus.append(review)
    return corpus

def train_and_dump_bow(corpus, bow_path):
    cv = CountVectorizer(max_features=1420)
    X = cv.fit_transform(corpus).toarray()
    pickle.dump(cv, open(bow_path, "wb"))
    return X

def train_test_and_dump_classifier(X, y, classifier_path):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)
    joblib.dump(classifier, classifier_path)
    y_pred = classifier.predict(X_test)
    print(accuracy_score(y_test, y_pred))

def current_version(out_path):
    return len(os.listdir(out_path)) + 1

if not os.path.exists(path_to_bow):
    os.makedirs(path_to_bow)

if not os.path.exists(path_to_classifier):
    os.makedirs(path_to_classifier)

dataset = load_data(path_to_tsv + tsv_name)
corpus = preprocess_data(dataset)
bow_name = "{}sentiment-model-{}".format(path_to_bow, current_version(path_to_bow))
X = train_and_dump_bow(corpus, bow_name)
y = dataset.iloc[:, -1].values
classifier_name = "{}classifier-model-{}".format(path_to_classifier, current_version(path_to_classifier))
train_test_and_dump_classifier(X, y, classifier_name)
