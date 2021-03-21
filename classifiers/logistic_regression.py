import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from .create_features import create_features_tfidf
from .create_features import create_features_vectorizer


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
        vec, x_train = create_features_vectorizer(features, x_train, ngramrange=ngrams)
    elif method == "tfidf":
        vec, x_train = create_features_tfidf(features, x_train, ngramrange=ngrams)
    else:
        print("Method has to be either count or tfidf")
        return 1


    LRparam_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l2'],
        'max_iter': list(range(100, 800, 100)),
        'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
    }
    LR = GridSearchCV(LogisticRegression(), param_grid=LRparam_grid, refit=True, verbose=3)
    # LR = LogisticRegression(max_iter=100, class_weight="balanced", C=1,solver='lbfgs',penalty='l2')
    model = LR.fit(x_train, y_train.values.ravel())
    print(model.best_params_)
    print(model.best_estimator_)
    return model, vec

