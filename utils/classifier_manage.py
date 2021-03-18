from classifiers import logistic_regression,svm
def choose_and_create_classifier(classifier, X_train, y_train, vectorizer,n_grams=(1,1)):
    """

    :param classifier:
    :param X_train:
    :param y_train:
    :param vectorizer:
    :param n_grams:
    :return:
    """

    if classifier == 'LOGISTIC REGRESSION':
        model,vectorizer = logistic_regression.setup_classifier(X_train,y_train,features='preprocessed',method=vectorizer,ngrams=n_grams)
    elif classifier == 'SVM':
        model, vectorizer = svm.setup_classifier(X_train, y_train, features='preprocessed',
                                                                 method=vectorizer, ngrams=n_grams)

    return model,vectorizer
