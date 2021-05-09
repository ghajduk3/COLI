import sklearn.svm as svm
from sklearn.model_selection import GridSearchCV
import pandas as pd

from .create_features import create_features_vectorizer
from .create_features import create_features_tfidf, combine_features


def setup_classifier(x_train:pd.DataFrame,y_train: pd.DataFrame ,features="preprocessed",method="count",ngrams=(1, 1)):
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
        vec, x_train, topic_model_dict = combine_features(features, x_train,method=method, ngramrange=ngrams)
    elif method == "tfidf":
        vec, x_train, topic_model_dict = combine_features(features, x_train,method=method,ngramrange=ngrams)
    else:
        print("Method has to be either count or tfidf")
        return 1
    param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf', 'poly', 'sigmoid']}
    # SVM= GridSearchCV(svm.SVC(), param_grid, refit=True, verbose=2,cv=2)
    SVM = svm.SVC(class_weight='balanced')
    model = SVM.fit(x_train, y_train.values.ravel())
    # print(model.best_params_)
    # print(model.best_estimator_)

    return model, vec, topic_model_dict
