import logging,sys
from preprocess import preparator
import re,math
import pandas as pd
from iteration_utilities import deepflatten
from pandas.core.common import flatten
logger = logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',level=logging.DEBUG)
from utils import pipeline

def transform_text(text,indexes):
    indexes = eval(indexes) if isinstance(indexes,str) else []
    comment_by_idx = []
    for index,comment in enumerate(re.split('\d.\s+',text)[1:]):
        if indexes and index+1 in indexes:
            comment_by_idx.append([comment,1])
        else:
            comment_by_idx.append([comment,0])
    return comment_by_idx

if __name__ == '__main__':
    # preparator.Preparator.prepare_all(sys.argv[1])
    # data = pd.read_csv(sys.argv[1])
    # data['text'] = data.apply(lambda row: transform_text(row.text,row.hate_speech_idx),axis=1)
    # comments = [item for sublist in data['text'].tolist() for item in sublist]
    # print(len(comments))
    # print(comments[:5])
    ds = pipeline.load_labeled_datasets()
    x,y = pipeline.run_dataset_preparation(ds)




