import pandas as pd
import gensim
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy import sparse


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

    ldamodel = gensim.models.ldamodel.LdaModel(data['corpus'].to_list(), num_topics=num_topics, id2word=dictionary, passes=5)
    # gensim.models.ldamodel.LdaModel.load()
    # ldamodel.save('model5.gensim')
    data['topics'] = data['corpus'].apply(lambda x: max(ldamodel.get_document_topics(x), key=lambda x: x[1])[0])
    # return pd.concat([data, pd.get_dummies(data['topics'], prefix='topics', dummy_na=True)], axis=1).drop(['topics'],axis=1).drop(['corpus'], axis=1)
    return (ldamodel,dictionary), pd.get_dummies(data['topics'], prefix='topics', dummy_na=False).reset_index().drop(['index'], axis=1)

def combine_features(columns, data, method='tfidf' ,ngramrange=(1,1), num_topics=10):
    if method == 'tfidf':
        vectorizer, training_features = create_features_tfidf(columns, data, ngramrange)
    elif method == 'count':
        vectorizer, training_features = create_features_vectorizer(columns, data, ngramrange)
    topic_model_dictionary, topic_features = create_features_topic_modelling(data,num_topics)

    features = sparse.hstack([training_features, sparse.csr_matrix(topic_features.values)])
    # features = pd.concat([pd.DataFrame(training_features.toarray()),topic_features], axis=1, ignore_index=True)
    return vectorizer, topic_model_dictionary, features

def combine_features_test(x, vectorizer, topic_model_dict):
    topic_model,topic_dictionary = topic_model_dict
    tfidf_features = vectorizer.transform(x['preprocessed'].values)
    topic_features = x['preprocessed'].apply(lambda row: topic_dictionary.doc2bow(row.split(" "))).apply(lambda row: max(topic_model.get_document_topics(row), key=lambda x: x[1])[0])
    topic_features = pd.get_dummies(topic_features, prefix='topics',dummy_na=False).reset_index().drop(['index'],axis=1)
    combined_features = sparse.hstack([tfidf_features, sparse.csr_matrix(topic_features.values)])
    # return pd.concat([pd.DataFrame(tfidf_features.toarray()),topic_features],axis=1,ignore_index=True)
    return combined_features