import sklearn.svm as svm
from .create_features import create_features_vectorizer
from .create_features import create_features_tfidf


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
    SVM = svm.SVC(kernel='linear', C=1, gamma=1)
    model = SVM.fit(x_train, y_train.values.ravel())

    return model, vec


def predict(model, x_testing):
    predictions = model.predict(x_testing)

    return predictions