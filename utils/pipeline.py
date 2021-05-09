import utils.utilities
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
from preprocess.combinator import read_combine_datasets, normalize_eng_dataset_5, normalize_eng_dataset_6, normalize_slo_dataset_2
from classifiers import bert, create_features


def prepare_labeled_datasets(lang = 'eng'):
    """
    Preprocesses and prepares unstructured source datasets into structured datasets. Each processed dataset has two columns [Text,Label].
    Label represents binary class [1-Hate speech, 0-Non-hate speech].
     """
    print(os.path.join('data', 'source_data', lang))
    preparator.Preparator.prepare_all(os.path.join('data', 'source_data', lang), lang)

def combine_binary_datasets(lang = 'eng'):
    """

    Arguments
    ----------
    lang

    Returns
    -------

    """
    base_data_path = os.path.join(rootpath.detect(), 'data', 'structured_data', lang, 'binary')
    output_data_path = os.path.join(rootpath.detect(), 'data', 'final_data', lang, 'binary' , 'data.csv')
    if os.path.exists(base_data_path):
        dataset_numbers = [int(name.split('_')[1]) for name in os.listdir(base_data_path)]
        dataset_lower_bound, dataset_upper_bound = min(dataset_numbers) , max(dataset_numbers)
        dataset = read_combine_datasets(base_data_path,(dataset_lower_bound,dataset_upper_bound),concatenate=True)
        dataset.dropna(inplace=True)
        dataset.drop_duplicates(subset=['Text', 'Label'], keep='first',inplace=True)
        dataset.reset_index(inplace=True, drop=True)
        utils.utilities.write_to_file_pd(dataset,output_data_path)

def combine_multiclass_datasets(lang = 'eng'):
    """

    Parameters
    ----------
    lang

    Returns
    -------

    """
    base_data_path = os.path.join(rootpath.detect(), 'data', 'structured_data', lang, 'multiclass')
    output_data_path = os.path.join(rootpath.detect(), 'data', 'final_data', lang, 'multiclass', 'data.csv')
    if os.path.exists(base_data_path):
        dataset_numbers = [int(name.split('_')[1]) for name in os.listdir(base_data_path)]
        dataset_lower_bound, dataset_upper_bound = min(dataset_numbers), max(dataset_numbers)

        if lang == "eng":
            dataset_5,dataset_6 = read_combine_datasets(base_data_path, (dataset_lower_bound, dataset_upper_bound), concatenate=False)
            dataset_5 = normalize_eng_dataset_5(dataset_5)
            dataset_6 = normalize_eng_dataset_6(dataset_6)
            dataset = pd.concat([dataset_5,dataset_6], axis=0, ignore_index=True)
        
        if lang == "slo":
            dataset_1,dataset_2 = read_combine_datasets(base_data_path, (dataset_lower_bound, dataset_upper_bound), concatenate=False)
            dataset_2 = normalize_slo_dataset_2(dataset_2)
            dataset = pd.concat([dataset_1,dataset_2], axis=0, ignore_index=True)

        dataset.dropna(inplace=True)
        dataset.drop_duplicates(subset=['Text', 'Label'], keep='first',inplace=True)
        dataset.reset_index(inplace=True, drop=True)
        utils.utilities.write_to_file_pd(dataset, output_data_path)

def load_binary_datasets(lang='eng')-> pd.DataFrame:
    input_base_path = os.path.join(rootpath.detect(), 'data', 'final_data', lang, 'binary', 'data.csv')
    try:
        df = pd.read_csv(input_base_path)
    except pd.errors.EmptyDataError:
        df = pd.DataFrame()
    return df

def load_multiclass_datasets(lang='eng')-> pd.DataFrame:
    input_base_path = os.path.join(rootpath.detect(), 'data', 'final_data', lang, 'multiclass', 'data.csv')
    try:
        df = pd.read_csv(input_base_path)
    except pd.errors.EmptyDataError:
        df = pd.DataFrame()
    return df

def create_final_datasets(lang = "eng"):
    combine_binary_datasets(lang)
    combine_multiclass_datasets(lang)

    binary = load_binary_datasets(lang)
    multiclass = load_multiclass_datasets(lang)

    if not binary.empty:
        binary_non_hate = binary.loc[binary["Label"] == 0]

        multiclass_full = pd.concat([multiclass, binary_non_hate], axis=0, ignore_index=True)

        multiclass_full.dropna(inplace=True)
        multiclass_full.drop_duplicates(subset=['Text', 'Label'], keep='first',inplace=True)
        multiclass_full.reset_index(inplace=True, drop=True)
        multiclass_output_data_path = os.path.join(rootpath.detect(), 'data', 'final_data', lang, 'multiclass', 'data.csv')
        utils.utilities.write_to_file_pd(multiclass_full, multiclass_output_data_path)
    
    if not multiclass.empty:
        multiclass_non_hate = multiclass.loc[multiclass["Label"] == 0]
        multiclass_hate = multiclass.loc[multiclass["Label"].isin([1, 2, 3, 4, 5])]
        multiclass_hate["Label"] = 1

        binary_full = pd.concat([binary, multiclass_non_hate, multiclass_hate], axis=0, ignore_index=True)

        binary_full.dropna(inplace=True)
        binary_full.drop_duplicates(subset=['Text', 'Label'], keep='first',inplace=True)
        binary_full.reset_index(inplace=True, drop=True)
        binary_output_data_path = os.path.join(rootpath.detect(), 'data', 'final_data', lang, 'binary' , 'data.csv')
        utils.utilities.write_to_file_pd(binary_full, binary_output_data_path)

def load_labeled_datasets(dataset_number=(1,5),lang='eng',type='binary',concatenate=True)->Union[pd.DataFrame,List[pd.DataFrame]]:
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
     lang                String
                         Language of the data to be loaded
     type                String
                         Type of the classification labels to be loaded. Default: binary, available are binary and multiclass
     Returns
     -------
     all_dataframes      Union[pd.DataFrame,List[pd.DataFrame]]
                         All datasets conctatenated into one pd.DataFrame or list of all datasets.
     """
    return read_combine_datasets(os.path.join(rootpath.detect(),'data', 'structured_data',lang,type),dataset_number,concatenate)

def run_dataset_preparation(dataset:pd.DataFrame, lang="eng", remove_stopwords=True, do_lemmatization=True)->Tuple[pd.DataFrame,pd.DataFrame]:
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
    if lang == "eng":
        dataset['preprocessed'] = dataset['Text'].apply(prep.eng_preprocessing, remove_stopwords=remove_stopwords, do_lemmatization=do_lemmatization)
    elif lang == "slo":
        dataset = prep.slo_preprocessing(dataset, remove_stopwords=remove_stopwords, do_lemmatization=do_lemmatization)

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
                         'LR', 'SVM', 'XGB'.

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
        model, vectorizer, topic_model_dict = choose_and_create_classifier(classifier, X_train.to_frame(), y_train.to_frame(), vectorizer)
        return model,vectorizer,topic_model_dict,X_test.to_frame(),y_test.to_frame()
    else:
        x_train,y_train = balance_data_over_sample(x_train,y_train)
        model, vectorizer, topic_model_dict = choose_and_create_classifier(classifier, x_train, y_train, vectorizer)
        return model,vectorizer,topic_model_dict

def transform_and_predict(model,vectorizer,topic_model_dict,x_test:pd.DataFrame,features='preprocessed')->pd.DataFrame:
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
    # x_test = vectorizer.transform(x_test[features].values)
    x_test = create_features.combine_features_test(x_test,vectorizer,topic_model_dict)
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
                         'LR', 'SVM', 'XGB'.

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

def run_bert_experiment(x_english,y_english,x_slovene,y_slovene,type='bi'):
    model = bert.setup_classifier(
        model_name="classifiers/bert/CroSloEngual",
        num_labels=2
    )

    dataset = bert.setup_data(
        model_name = "classifiers/bert/CroSloEngual",
        x = x_english,
        y = y_english,
        do_lower_case = False,
        max_length = 180
    )
    model_name = 'binary.pt' if type == 'bi' else 'multiclass.pt'

    model.load_state_dict(bert.load_model("models/bert/" + model_name))

    predictions, true_labels = bert.test_classifier(
        model = model,
        dataset = dataset,
        batch_size = 32
    )
    utils.utilities.print_performance_metrics(predictions, true_labels)

    dataset_slovene = bert.setup_data(
        model_name = "classifiers/bert/CroSloEngual",
        x = x_slovene,
        y = y_slovene,
        do_lower_case = False,
        max_length = 180
    )
    predictions_slov, true_labels_slov = bert.test_classifier(
        model = model,
        dataset = dataset,
        batch_size = 32
    )
    utils.utilities.print_performance_metrics(predictions_slov, true_labels_slov)




