import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from typing import List,AnyStr
from classifiers import create_features
from .classifier_manage import choose_and_create_classifier
from .dataset_balancer import balance_dataset
from .utilities import write_to_file_pd
from .utilities import print_performance_metrics


def generate_evaluation_report(y_true:pd.DataFrame,y_pred:pd.DataFrame,target_names:List[AnyStr]):
    """
     Generate text report showing the main classification metrics.
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
    print(f1_score(y_true,y_pred,average='weighted'))
    print(accuracy_score(y_true,y_pred))
    return classification_report(y_true,y_pred,target_names=target_names)

def generate_evaluation_report_cv(classifier,vectorizer,x_data,y_data,features='preprocessed'):
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
    f1 = []
    acc = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    trues = []
    preds = []

    for train_index, test_index in skf.split(x_data, y_data):
        X_train, X_test = x_data.iloc[train_index].to_frame(), x_data.iloc[test_index].to_frame()
        y_train, y_test = y_data.iloc[train_index].to_frame(), y_data.iloc[test_index].to_frame()
        # Balances training data
        # X_train,y_train = balance_dataset(X_train,y_train,method='OVER')
        model, vector, topic_model_dict = choose_and_create_classifier(classifier, X_train, y_train, vectorizer)
        # X_test = vector.transform(X_test[features].values)
        X_test = create_features.combine_features_test(X_test,vector,topic_model_dict)
        y_pred = model.predict(X_test)
        trues.extend(y_test['Label'].tolist())
        preds.extend(y_pred.tolist())
        #f1.append(f1_score(y_test,y_pred,average='weighted'))
        #acc.append(accuracy_score(y_test,y_pred))
    #print("{:<35}:{:>8.4f}".format("Weighted f1", round(sum(f1)/len(f1), 4)))
    #print("{:<35}:{:>8.4f}".format("Accuracy", round(sum(acc)/len(acc), 4)))
    #return sum(f1)/len(f1),sum(acc)/len(acc)
    print_performance_metrics(preds, trues)
    return None

def data_label_counts(balanced,imbalanced):
    """
     Explores count of distinct class labels from each balanced and imbalaced dataset.
     Creates and saves the report into a csv file stored into data/reports.
     Arguments
     ----------
     balanced            List[pd.DataFrame]
                         List of balanced input train data sets.

     imbalanced          List[pd.DataFrame]
                         List of imputbalanced input train data sets.
     Returns
     -------
     label_counts        pd.DataFrame
                         Label counts accross all balanced/imbalanced datasets.
     """
    label_counts = pd.DataFrame(columns=[])
    for index,dataset in enumerate(balanced):
        name = 'Balanced dataset_'+str(index+1)
        label_counts[name] = dataset['Label'].value_counts()
    for index,dataset in enumerate(imbalanced):
        name = 'Imbalanced dataset_'+str(index+1)
        label_counts[name] = dataset['Label'].value_counts()
    return label_counts

# def data_word_counts(imbalanced : List[pd.DataFrame]):
#     for index, dataset in enumerate(imbalanced):
#         dataset = dataset.dropna(how='any', axis=0)
#         dataset = dataset['Text']
#         for row in dataset:
#             print(row,type(row))

def generate_data_exploration_report(balanced: List[pd.DataFrame], imbalanced: List[pd.DataFrame]):
    """
     Generates data exploration report.
     Arguments
     ----------
     balanced            List[pd.DataFrame]
                         List of balanced input train data sets.

     imbalanced          List[pd.DataFrame]
                         List of imputbalanced input train data sets.

     """
    pass
    # data_word_counts(imbalanced)
    # label_counts = data_label_counts(balanced,imbalanced)
    # write_to_file_pd(label_counts,'./data/reports/label_value_counts.csv')
