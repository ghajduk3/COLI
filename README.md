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

<a name="usage"></a>
### Usage

The project is structured to implement multiple classifiers for two classification tasks, a binary and multiclass.
In order to reproduce results from the report a `CLI` application has been implemented.  Following instructions assume that the user is in project's root.

1. In order to run CLI application with help description issue:
```shell script
python main.py --help
```



