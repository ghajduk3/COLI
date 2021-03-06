import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from .create_features import create_features_tfidf, combine_features
from .create_features import create_features_vectorizer
from imblearn.over_sampling import RandomOverSampler,SMOTE


def setup_classifier(x_train: pd.DataFrame, y_train: pd.DataFrame, features="preprocessed", method="count", ngrams=(1, 1)):
    """
    Finds out best parameter combination for sklearn implementation of Logistic regression using GridSearch. Returns trained model and vectorizer.
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
        vec, topic_model_dict, x_train = combine_features(features, x_train,method='count',ngramrange=ngrams)
    elif method == "tfidf":
        vec, topic_model_dict, x_train = combine_features(features, x_train,method='tfidf',ngramrange=ngrams)
    else:
        print("Method has to be either count or tfidf")
        return 1
    LRparam_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l2'],
        'max_iter': list(range(100, 800, 100)),
        'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
    }
    # LR = GridSearchCV(LogisticRegression(class_weight='balanced'), param_grid=LRparam_grid, refit=True, verbose=3)
    LR = LogisticRegression(solver='lbfgs',class_weight='balanced',max_iter=5000)
    model = LR.fit(x_train, y_train.values.ravel())

    return model, vec, topic_model_dict

