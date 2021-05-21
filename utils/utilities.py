import os, csv
import logging,re
import rootpath, pandas as pd
from typing import AnyStr, Tuple, List
from sklearn import metrics
import numpy as np

logger = logging.getLogger(__name__)


def read_file_contents(input_file_path: AnyStr, delimiter=",", encoding="utf-8") -> Tuple:
    """
    Reads content and class labels from  csv file.
    ----------
    Parameters
    input_file_path     AnyStr
                        Input file path
    Returns
    ----------
    content             Tuple
                        Class labels and input data.
    """
    try:
        with open(input_file_path,'r',encoding=encoding) as csv_file:
            reader = csv.reader(csv_file,delimiter=delimiter)
            labels = next(reader)
            data = [line for line in reader]
            logger.info("Data has been read succesfully")
            return labels, data
    except IOError:
        logger.exception("I/O error")
    except FileNotFoundError:
        logger.exception("FileNotFound error")
    except Exception:
        logger.exception("Unexpected error")

def read_file_contents_pd(input_file_path:AnyStr)->pd.DataFrame:
    """
    Reads content from  csv file.
    ----------
    Parameters
    input_file_path     AnyStr
                        Input file path
    Returns
    ----------
    content             pd.DataFrame
                        File contents.
    """

    return pd.read_csv(input_file_path)

def write_to_file_pd(content : pd.DataFrame ,output_file_path: AnyStr):
    """
    Writes content to csv file.
    ----------
    Parameters
    content             pd.DataFrame
                        Input data.

    output_file_path    AnyStr
                        Output file path.
    """
    content.to_csv(output_file_path,index=False)

def read_file_contents_txt(input_file_path:str):
    try:
        with open(input_file_path, 'r', encoding='utf-8') as txt_file:
            return txt_file.readline()
    except IOError:
        logger.exception("I/O error")
    except FileNotFoundError:
        logger.exception("FileNotFound error")
    except Exception:
        logger.exception("Unexpected error")

def write_to_file(content: List, output_file_path:AnyStr):
    """
    Writes content to csv file.
    ----------
    Parameters
    content             List
                        Input data.

    output_file_path    AnyStr
                        Output file path.
    """
    try:
        with open(output_file_path,'w+',newline='',encoding='utf-8') as output_file:
            writer = csv.writer(output_file)
            writer.writerows(content)
            logger.info("Data has been succesfully written to file")
    except IOError:
        logger.exception("I/O error")
    except FileNotFoundError:
        logger.exception("FileNotFound error")
    except Exception:
        logger.exception("Unexpected error")

def split_input_path(input_path:AnyStr)->Tuple[AnyStr,AnyStr]:
    '''
    Splits absolute file path to the directory path and the filename.ext.
    ----------
    Parameters
    input_path          AnyStr
                        Absolute or relative path.
    --------
    Returns
    dir_name            AnyStr
                        Path to the directory of the file

    file_name           AnyStr
                        File name.
    '''
    path = os.path.normpath(input_path)
    return (os.path.dirname(path),os.path.basename(path))

def transform_dataset_2_3(text:str,indexes:str)->list:
    indexes = eval(indexes) if isinstance(indexes,str) else []
    comment_by_idx = []
    for index,comment in enumerate(re.split('^(\d+).\s+',text,flags=re.MULTILINE)[1:][1::2]):
        if indexes and index+1 in indexes:
            comment_by_idx.append([comment,1])
        else:
            comment_by_idx.append([comment,0])
    return comment_by_idx

def strip_and_replace_new_lines(data:list)->list:
    return [[str(col).replace('\n', ' ').replace('\r', ' ').strip() for col in row] for row in data]

def class_distribution_dataset(path:str)->dict:
    labels, data = read_file_contents(path, ',')

    class_distribution = {}

    for row in data:
        if row[1] not in class_distribution:
            class_distribution[row[1]] = 0
        class_distribution[row[1]] += 1

    return class_distribution


def avg_accuracy(predictions,labels):
    return sum(1 for i, j in zip(predictions, labels) if i == j) / len(predictions) 

def round_to_percentages(value):
    return round(round(value, 4) * 100, 2)

def majority_class(data):
    elements_count = {}
    for element in data:
        if element not in elements_count:
            elements_count[element] = 0
        elements_count[element] += 1
    maximum = max(elements_count, key=elements_count.get)
    return elements_count[maximum] / len(data)

def print_performance_metrics(predicted, true):
    matrix = {}
    labels = sorted(list(set(true)))

    for t in labels:
        for p in labels:
            matrix[(t, p)] = sum([1 for i, j in zip(true, predicted) if i == t and j == p]) / len(true)

    print()
    print("Confusion Matrix")
    print("Rows - Actual")
    print("Columns - Predicted")
    print()

    print(("{:>3}"+" | "+"{:>6}"*len(labels)+" | ").format(" ", *[p for p in labels]))

    print("-" * (3 + 3 + 6 * len(labels) + 3 + 6))

    for t in labels:
        print(("{:>3}"+" | "+"{:>6.2f}"*len(labels)+" | "+"{:>6.2f}").format(t, *[round_to_percentages(matrix[(t, p)]) for p in labels], round_to_percentages(sum([matrix[(t, p)] for p in labels]))))

    print("-" * (3 + 3 + 6 * len(labels) + 3 + 6))

    print(("{:>3}"+" | "+"{:>6.2f}"*len(labels)+" | ").format(" ", *[round_to_percentages(sum([matrix[(t, p)] for t in labels])) for p in labels]))
    
    print()
    
    y_true = np.array(true)
    y_pred = np.array(predicted)

    print(metrics.classification_report(y_true, y_pred))