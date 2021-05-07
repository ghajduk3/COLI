import pandas as pd
import gensim
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def create_features_vectorizer(columns, data, ngramrange=(1, 1)):
    # intialise Countvectorizer and fit transform to data
    vectorizer = CountVectorizer(ngram_range=ngramrange)
    vectorizer.fit_transform(data[columns].values)

    # build matrixes for training_features and testing_features
    training_features = vectorizer.transform(data[columns].values)

    return vectorizer, training_features


def create_features_tfidf(columns, data, ngramrange=(1, 1)):

    # intialise Tfidfvectorizer and fit transform to data
    tfidf_vectorizer = TfidfVectorizer(ngram_range=ngramrange)
    tfidf_vectorizer.fit_transform(data[columns].values)

    # build matrixes for training_features and testing_features
    training_features = tfidf_vectorizer.transform(data[columns].values)

    return tfidf_vectorizer, training_features

def create_features_topic_modelling(data,num_topics=10):
    data['corpus'] = data['preprocessed'].apply(lambda x:x.split(" "))
    dictionary = gensim.corpora.Dictionary(data['corpus'].to_list())
    data['corpus'] = data['corpus'].apply(lambda x: dictionary.doc2bow(x))

    ldamodel = gensim.models.ldamodel.LdaModel(data['corpus'].to_list(), num_topics=num_topics, id2word=dictionary, passes=15)
    ldamodel.save('model5.gensim')
