import pandas as pd
import xgboost as xgb
from .create_features import create_features_vectorizer
from .create_features import create_features_tfidf, combine_features
import pickle
import os


def setup_classifier(x_train, y_train,features="preprocessed", method="count", ngrams=(1, 1)):
    """
    Finds out best parameter combination for sklearn implementation of SVM using GridSearch. Returns trained model and vectorizer.
    Arguments
    ----------
    x_train  	        pd.DataFrame
                    	Input training data for the classifier

    y_train     	    Pandas dataframe
                    	The dataframe containing the y training data for the classifier

    features         	AnyStr
                    	Names of columns of df that are used for trainig the classifier

    method               AnyStr
                         Name of a preferred vectorizer to be used. Currently available :
                         'tfidf', 'count' vectorizer.

    ngrams:             Tuple
                        (min_n, max_n), with min_n, max_n integer values
                        range for ngrams used for vectorization

    Returns
    -------
    model		        sklearn LogisticRegression Model
            			Trained LogistciRegression Model
    vec          	    sklearn CountVectorizer or TfidfVectorizer
                    	CountVectorizer or TfidfVectorizer fit and transformed for training data
    """

    if method == "count":
        vec, topic_model_dict, x_train = combine_features(features, x_train, method='count', ngramrange=ngrams)
    elif method == "tfidf":
        vec, topic_model_dict, x_train = combine_features(features, x_train, method='tfidf', ngramrange=ngrams)
    else:
        print("Method has to be either count or tfidf")
        return 1

    xgboost_classifier = xgb.XGBClassifier(learning_rate = 0.05, n_estimators=300, max_depth=5,random_state=42)
    model = xgboost_classifier.fit(x_train, y_train.values.ravel())
    return model, vec, topic_model_dict
