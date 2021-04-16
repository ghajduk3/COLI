from preprocess import preparator
from typing import Union,List,Tuple,AnyStr
import os,rootpath
import pandas as pd
from .dataset_balancer import balance_dataset, balance_data_over_sample,split_dataset
import preprocess.preprocess as prep
from .classifier_manage import choose_and_create_classifier
from .exploration_evaluation import generate_evaluation_report,generate_data_exploration_report,generate_evaluation_report_cv
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

def prepare_labeled_datasets():
    """
    Preprocesses and prepares unstructured source datasets into structured datasets. Each processed dataset has two columns [Text,Label].
    Label represents binary class [1-Hate speech, 0-Non-hate speech].
     """
    preparator.Preparator.prepare_all('../source_data/eng')


def load_labeled_datasets(dataset_number=(1,5),concatenate=True)->Union[pd.DataFrame,List[pd.DataFrame]]:
    """
     Loads specific number of structured data set files from csv files located in /source_data/dataset_{d+} where
     dataset_number denotes lower and upper bound of datasets that are to be loaded. I.e dataset_number(1,5) loads datasets 1-5.
     Loaded datasets are concatenated and returned as one pandas dataframe if concatenate is set to True. If not list containing
     all loaded datasets is returned.
     Arguments
     ----------
     dataset_number      Tuple
                         Represents start and end number of labeled datasets to be loaded.
                         I.e dataset_number(2,4) denotes that datasets 2,3,4 are to be loaded.
     concatenate         Bool
                         Flag which indicates wheter all dataframes are to be concatenated into one.
     Returns
     -------
     all_dataframes      Union[pd.DataFrame,List[pd.DataFrame]]
                         All datasets conctatenated into one pd.DataFrame or list of all datasets.
     """
    structured_data_base = os.path.join(rootpath.detect(), 'structured_data', 'dataset_')
    all_dataframes = []
    first_ds, last_ds = dataset_number
    for index in range(first_ds,last_ds+1):
        dataset_path = os.path.join(structured_data_base+str(index),'data.csv')
        df = pd.read_csv(dataset_path, index_col=None)
        print(dataset_path)
        print(len(df['Label'].to_list()))
        all_dataframes.append(df)
    if concatenate:
        all_dataframes = pd.concat(all_dataframes, axis=0, ignore_index=True)
        return all_dataframes
    else:
        return all_dataframes

def run_dataset_preparation(dataset:pd.DataFrame)->Tuple[pd.DataFrame,pd.DataFrame]:
    """
     Removes all null rows. Executes preprocessing on the provided dataset.
     Returns preprocessed data and its class labels.
     Arguments
     ----------
     dataset             pd.DataFrame
                         Input dataset consisting of Text and Label columns.
     Returns
     -------
     data                Tuple[pd.DataFrame,pd.DataFrame]
                         Preprocessed and (balanced) text data and its labels.
     """

    dataset = dataset.dropna(how='any', axis=0)
    dataset['preprocessed'] = dataset['Text'].apply(prep.preprocessing)
    x,y = dataset['preprocessed'], dataset['Label']
    # if balance:
    #     x,y = balance_data_over_sample(x,y)
    data = (x,y)
    return data

def train_and_split(classifier:AnyStr, vectorizer:AnyStr, x_train:pd.DataFrame, y_train:pd.DataFrame,split=True,method='UNDER')->Tuple[object,object]:
    """
     Splits inpur x,y dataset into train and test datasets if split is set to true.
     Balances training dataset through under-sampling or over-sampling.
     Chooses and trains selected classifier model with features created by selected vectorizer.
     Arguments
     ----------
     classifier          AnyStr
                         Name of a preferred classifier to be used. Currently available :
                         'LOGISTIC REGRESSION', 'SVM', 'XGB'.

     vectorizer          AnyStr
                         Name of a preferred vectorizer to be used. Currently available :
                         'tfidf', 'count' vectorizer.

     x_train              pd.DataFrame
                          Input training data.

     y_train              pd.DataFrame
                          Class labels of the training data.

     split                bool
                          Flag which indicates whether the input data is to be splitted.

     balance              bool
                          Flag which indicates whether the input data is to be balanced.
     Returns
     -------
     model		          Model of implemented Classifiers
                          Trained Model

     vectorizer        	  sklearn CountVectorizer or TfidfVectorizer
                          CountVectorizer or TfidfVectorizer fit and transformed for training data
     """
    if split:
        class_weights = compute_class_weight('balanced', np.unique(y_train), y_train)
        # print(class_weights)
        X_train, X_test, y_train, y_test = split_dataset(x_train,y_train)
        # X_train,y_train = balance_dataset(X_train,y_train,method=method)
        # print(y_train['Label'].value_counts())
        model, vectorizer = choose_and_create_classifier(classifier, X_train.to_frame(), y_train.to_frame(), vectorizer)
        return model,vectorizer,X_test.to_frame(),y_test.to_frame()
    else:
        x_train,y_train = balance_data_over_sample(x_train,y_train)
        model, vectorizer = choose_and_create_classifier(classifier, x_train, y_train, vectorizer)
        return model,vectorizer

def transform_and_predict(model,vectorizer,x_test:pd.DataFrame,features='preprocessed')->pd.DataFrame:
    """
     Creates a feature set for the input test data. Predicts the class labels for input test data with the provided model.
     Arguments
     ----------
     model               Object
                         Trained model used for predicting labels based on test data.

     vectorizer          Object
                         Trained vectorizer model used for creating feature sets from test data.

     x_test               pd.DataFrame
                          Input test data.

     features             AnyStr
                          Labels of columns to be transformed into features.

     Returns
     -------
     y_pred               pd.DataFrame
                          Predicted class labels for input test data.
     """
    x_test = vectorizer.transform(x_test[features].values)
    y_pred = model.predict(x_test)
    return y_pred

def evaluate(y_true:pd.DataFrame, y_predicted:pd.DataFrame, target_names:List[AnyStr])->AnyStr:
    """
     Generates evaluation report based on true and predicted class labels.
     Arguments
     ----------
     y_true              pd.DataFrame
                         Real class labels.

     y_predicted         pd.DataFrame
                         Class labels predicted by selected model.

     target_names         List[AnyStr,AnyStr]
                          Target category names [Non-hate,Hate].

     Returns
     -------
     classification_report   classification_report
                          Text report showing the main classification metrics.
     """
    classification_report =  generate_evaluation_report(y_true, y_predicted, target_names)
    return classification_report

def evaluate_cross_validation(classifier:AnyStr,vectorizer:AnyStr,x_data:pd.DataFrame,y_data:pd.DataFrame)->Tuple[AnyStr,AnyStr]:
    """
     Performs a KFold cross validation based on specific selector and vectorizer.
     Generates cross validation report consisted of average f1 and accuracy score across all k validation passes.
     Arguments
     ----------
     classifier          AnyStr
                         Name of a preferred classifier to be used. Currently available :
                         'LOGISTIC REGRESSION', 'SVM', 'XGB'.

     vectorizer          AnyStr
                         Name of a preferred vectorizer to be used. Currently available :
                         'tfidf', 'count' vectorizer.

     x_data              pd.DataFrame
                         Input train data.

     y_data              pd.DataFrame
                         Class labels for input train data

     Returns
     -------
     classification_report   Tuple[AnyStr, AnyStr]
                          Tuple consisted of avg f1 and avf accuracy score.
     """
    classification_report = generate_evaluation_report_cv(classifier,vectorizer,x_data,y_data)
    return classification_report

def explore():
    """
     Explores count of distinct class labels from each balanced and imbalaced dataset.
     Creates and saves the report into a csv file stored into data/reports.
     """
    datasets = load_labeled_datasets((1,5),concatenate=False)
    balanced = [run_dataset_preparation(dataset,balance=True)[1] for dataset in datasets]
    imbalanced = [run_dataset_preparation(dataset, balance=False)[1] for dataset in datasets]
    generate_data_exploration_report(balanced,imbalanced)
