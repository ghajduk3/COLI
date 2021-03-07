import os, csv
import logging,re
import rootpath, pandas as pd

logger = logging.getLogger(__name__)


def read_file_contents(input_file_path: str, delimiter=",") -> tuple:
    """

    :param file_path:
    :param delimiter:
    :return: labels,data
    """
    try:
        with open(input_file_path,'r') as csv_file:
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

def read_file_contents_pd(input_file_path:str):
    """
    :param input_file_path:
    :return:
    """
    return pd.read_csv(input_file_path)

def read_file_contents_txt(input_file_path:str):
    try:
        with open(input_file_path, 'r') as txt_file:
            return txt_file.readline()
    except IOError:
        logger.exception("I/O error")
    except FileNotFoundError:
        logger.exception("FileNotFound error")
    except Exception:
        logger.exception("Unexpected error")

def write_to_file(content,output_file_path:str):
    """

    :param content:
    :param output_file_path:
    :return:
    """
    try:
        with open(output_file_path,'w+') as output_file:
            writer = csv.writer(output_file)
            writer.writerows(content)
            logger.info("Data has been succesfully written to file")
    except IOError:
        logger.exception("I/O error")
    except FileNotFoundError:
        logger.exception("FileNotFound error")
    except Exception:
        logger.exception("Unexpected error")

def split_input_path(input_path:str)->tuple:
    '''
    Splits absolute file path to the directory path and the filename.ext.
            Parameters:
                    input_path (str): Absolute or relative path.
            Returns:
                    dir_name (str): Path to the directory of the file
                    file_name (str): File name
    '''
    path = os.path.normpath(input_path)
    return (os.path.dirname(path),os.path.basename(path))

def transform_dataset_2_3(text:str,indexes:str)->list:
    indexes = eval(indexes) if isinstance(indexes,str) else []
    comment_by_idx = []
    for index,comment in enumerate(re.split('\d.\s+',text)[1:]):
        if indexes and index+1 in indexes:
            comment_by_idx.append([comment,1])
        else:
            comment_by_idx.append([comment,0])
    return comment_by_idx
