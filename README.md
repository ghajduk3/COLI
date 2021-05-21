# NLP - Cross lingual offensive language identification
##### authors: Gojko Hajdukovic, Simon Dimc, 05.2021

Table of contents:
1. [Setup](#setup)
2. [Usage](#usage)


<a name="setup"></a>
### Setup
These instructions assume that the user is in repo's root.
```shell script
cd <repo_root>
```

1. In order to set-up virtual environment issue:
```shell script
python -m venv venv
#Activate the environment
source venv/bin/activate
```
2. To install all project related dependencies issue:
```shell script
pip3 install -r requirements.txt
python -m spacy download en_core_web_sm
```

3. Get datasets:
Datasets are in folder data/source_data. Get datasets from following sources and put them into folders:

    - data/source_data/eng/binary/dataset_1
    source: https://github.com/sjtuprog/fox-news-comments
    You will have to parse the json file fox-news-comments.json and convert it into a data.csv file with format: Label:Text.
    - data/source_data/eng/binary/dataset_2
    source: https://github.com/jing-qian/A-Benchmark-Dataset-for-Learning-to-Intervene-in-Online-Hate-Speech
    reddit data
    Rename file to data.csv.
    - data/source_data/eng/binary/dataset_3
    source: https://github.com/jing-qian/A-Benchmark-Dataset-for-Learning-to-Intervene-in-Online-Hate-Speech
    gab data
    Rename file to data.csv.
    - data/source_data/eng/binary/dataset_4
    source: https://github.com/Vicomtech/hate-speech-dataset
    Copy following folders and files.
    folders: all_files, sampled_test
    files: annotations_metadata.csv
    - data/source_data/eng/multiclass/dataset_5
    source: https://github.com/mayelsherif/hate_speech_icwsm18
    You will have to either download the tweets using the provided Tweet IDs or contact the authors. Put the tweets in csv files in format tweet_id,tweet, inside a downloaded_tweets_dataset folder. Name of the csv files should be the same as in the provided filenames with Tweet Ids.
    - data/source_data/eng/multiclass/dataset_6
    source: https://github.com/Mrezvan94/Harassment-Corpus
    You will have to contact the authors. Put the csv files inside a tweets_dataset folder.
    - data/source_data/slo/multiclass/dataset_2
    source: https://www.clarin.si/repository/xmlui/handle/11356/1398
    You will have to either download the tweets using the provided Tweet IDs or contact the authors. You will have to parse the data into a data.csv file with format: Text,Class,Type.
    

<a name="usage"></a>
### Usage

The project is structured to implement multiple classifiers for two classification tasks, a binary and multiclass.
In order to reproduce results from the report a `CLI` application has been implemented.  Following instructions assume that the user is in project's root.

1. In order to run CLI application with help description issue:
```shell script
python main.py --help
```

2. Examples:
```shell script
python main.py --prepareData true --type multi --model LR
python main.py -pd false -t bi -m BERT
```



