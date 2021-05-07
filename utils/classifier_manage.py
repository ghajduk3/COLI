from classifiers import logistic_regression,svm,xg_boost
import pickle
import pandas as pd
import os,rootpath
from datetime import date
from typing import AnyStr,Tuple
def choose_and_create_classifier(classifier: AnyStr, X_train: pd.DataFrame, y_train:pd.DataFrame, vectorizer:AnyStr,n_grams=(1,1))->Tuple[object,object]:
    """
    Creates classifier for the given input training data and parameters.
    Arguments
    ----------
    classifier          AnyStr
                        Name of a preferred classifier to be used. Currently available :
                        'LOGISTIC REGRESSION', 'SVM', 'XGB'.

    X_train	            pd.DataFrame
                        Features to train classifier

    y_data              pd.DataFrame
                        Labels to train classifier

    vectorizer          AnyStr
                        Name of a preferred vectorizer to be used. Currently available :
                        'tfidf', 'count' vectorizer.

    n_grams             Tuple
                        (min_n, max_n), with min_n, max_n integer values
                        range for ngrams used for vectorization
    Returns
    ----------
    model		        Model of implemented Classifiers
            			Trained Model

    vectorizer        	sklearn CountVectorizer or TfidfVectorizer
                    	CountVectorizer or TfidfVectorizer fit and transformed for training data

    """
    if classifier == 'LR':
        model,vectorizer = logistic_regression.setup_classifier(X_train,y_train,features='preprocessed',method=vectorizer,ngrams=n_grams)
    elif classifier == 'SVM':
        model, vectorizer = svm.setup_classifier(X_train, y_train, features='preprocessed',method=vectorizer, ngrams=n_grams)
    elif classifier == 'XGB':
        model, vectorizer = xg_boost.setup_classifier(X_train, y_train, features='preprocessed', method=vectorizer,ngrams=n_grams)



    return model,vectorizer


def save_classifier(model,name,vec):
    """
    Saves classifier to subfolder models in current working directory.

    Arguments
    ----------
    model		        Model of implemented Classifiers
            			Trained Model
    vec        	    sklearn CountVectorizer or TfidfVectorizer
                    	CountVectorizer or TfidfVectorizer fit and transformed for training data
    """
    model_name = "model_" + name + "_" + str(date.today()) + ".pkl"
    vec_name = "vec_" + name + "_" + str(date.today()) + ".pkl"
    model_path = os.path.join(rootpath.detect(), 'models', model_name)
    vec_path = os.path.join(rootpath.detect(), 'models', vec_name)

    pickle.dump(model, open(model_path, "wb"))
    pickle.dump(vec, open(vec_path, "wb"))


def load_classifier(model_path: AnyStr, vec_path: AnyStr)->Tuple[object, object]:
    """
    Loads train model and vectorizer from input paths.

    Arguments
    ----------
    model_path	        AnyStr
                        The path where the classifier is stored.

    vec_path:	        AnyStr
                        The path where the vectorizer is stored
    Returns
    ----------
    model		        Model of implemented Classifiers
            			Trained Model
    vec         	    sklearn CountVectorizer or TfidfVectorizer
                    	CountVectorizer or TfidfVectorizer fit and transformed for training data
    """
    model = pickle.load(open(model_path, "rb"))
    vec = pickle.load(open(vec_path, "rb"))

    return model, vec