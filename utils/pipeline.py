from preprocess import preparator
import os,rootpath
import pandas as pd
from utils import constants,dataset_balancer
import preprocess.preprocess as prep

def prepare_labeled_datasets():
    preparator.Preparator.prepare_all('../source_data/')


def load_labeled_datasets(num_datasets=1,concatenate=False):
    structured_data_base = os.path.join(rootpath.detect(), 'structured_data', 'dataset_')
    li = []
    for index in range(1,num_datasets+1):
        dataset_path = os.path.join(structured_data_base+str(index),'data.csv')
        df = pd.read_csv(dataset_path, index_col=None)
        li.append(df)
    return pd.concat(li, axis=0, ignore_index=True)

def run_dataset_preparation(dataset):
    #Preprocess and balance dataset

    dataset['preprocessed'] = dataset['Text'].apply(prep.preprocessing)
    x_balanced , y_balanced = dataset_balancer.balance_data(dataset['Text'],dataset['Label'])
    return x_balanced,y_balanced

# Splits and trains dataset
def train_and_split(classifier, vectorizer, n_grams, x_data, y_data,split=True):
    # returns trained model, x_test,y_test

    if split:
        X_train, X_test, y_train, y_test = dataset_balancer.split_dataset(x_data,y_data)

    model, vec, x_test = choose_and_create_classifier(classifier, x_train, y_train, df[['preprocessed']], vectorizer,
                                                      ngrams)

