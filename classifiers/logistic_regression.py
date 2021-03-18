import pandas as pd
from sklearn.linear_model import LogisticRegression
from .create_features import create_features_vectorizer
from .create_features import create_features_tfidf
import pickle
import os


def setup_classifier(x_train,
                             y_train,
                             features="preprocessed",
                             method="count",
                             ngrams=(1, 1)
                             ):

    # generate x and y training data

    if method == "count":
        vec, x_train = create_features_vectorizer(features, x_train, ngramrange=ngrams)
    elif method == "tfidf":
        vec, x_train = create_features_tfidf(features, x_train, ngramrange=ngrams)
    else:
        print("Method has to be either count or tfidf")
        return 1

    # train classifier
    log_reg_classifier = LogisticRegression(max_iter=1000, class_weight="balanced",)
    model = log_reg_classifier.fit(x_train, y_train.values.ravel())

    return model, vec


def predict(model, X_testing):
    predictions = model.predict(X_testing)

    return predictions