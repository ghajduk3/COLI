import spacy
import re
import classla
import nltk

from nltk.stem import PorterStemmer,SnowballStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

classla.download('sl')

nlp_eng = spacy.load("en_core_web_sm", disable=['tagger', 'parser', 'ner'])
eng_stopwords = set(nlp_eng.Defaults.stop_words)
slo_stopwords = set(stopwords.words('slovene'))

def eng_preprocessing(text, remove_stopwords=True, do_lemmatization=True):
    """
    Applies different steps of preprocessing to a text.
    Preprocessing includes:
    - remove standard stopwords (english stopwords from spacy)
    - perform lemmatizing

    Arguments
    ----------
    text:                   AnyStr
                            Text which should be converted by preprocessing
    Returns
    -------
    preprocessed_text:      String
                            Text which is converted by preprocessing.
    """
    
    text = base_preprocessing(text)

    tokens = []

    # split text to single words
    words = word_tokenize(text)

    lemmer = WordNetLemmatizer()

    # remove stopwords and words with length 1
    for word in words:
        if not remove_stopwords or word not in eng_stopwords:
            if do_lemmatization:
                word = lemmer.lemmatize(word)
            tokens.append(word)

    # convert tokens back to text
    preprocessed_text = ' '.join([str(element) for element in tokens])
    return preprocessed_text

def slo_preprocessing(dataset, remove_stopwords=True, do_lemmatization=True):

    # do base proccesing
    dataset['preprocessed'] = dataset['Text'].apply(base_preprocessing)

    # create pipelines
    tokenizer = classla.Pipeline('sl', processors='tokenize', type='nonstandard', logging_level='WARN')
    lemmatizer = classla.Pipeline('sl', processors='tokenize, lemma', type='nonstandard', logging_level='WARN')

    # do tokenization
    documents = '\n'.join(dataset['preprocessed'].values)
    out_docs = tokenizer(documents)

    for i, sentence in enumerate(out_docs.sentences):
        #print("DOCUMENT")
        seq = []
        for word in sentence.words:
            if not remove_stopwords or word.text not in slo_stopwords:
                seq.append(word.text)

        dataset.at[i, 'preprocessed'] = ' '.join(seq)

    # do lemmatization
    if do_lemmatization:
        documents = '\n'.join(dataset['preprocessed'].values)
        out_docs = lemmatizer(documents)
        
        for i, sentence in enumerate(out_docs.sentences):
            dataset.at[i, 'preprocessed'] = ' '.join(word.lemma for word in sentence.words)

    return dataset

def base_preprocessing(text):
    """
    Applies different steps of preprocessing to a text.
    Preprocessing includes:
    - remove all emoticons
    - remove non-standard lexical tokens (which are not numeric or alphabetical)
    - remove url and @name mentions
    - convert all letters to lower case

    Arguments
    ----------
    text:                   AnyStr
                            Text which should be converted by preprocessing
    Returns
    -------
    preprocessed_text:      String
                            Text which is converted by preprocessing.
    """
    EMOJI_PATTERN = re.compile(
        "(["
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F700-\U0001F77F"  # alchemical symbols
        "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
        "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        "\U0001FA00-\U0001FA6F"  # Chess Symbols
        "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        "])"
    )
    text = re.sub(EMOJI_PATTERN,"",text)
    # remove (twitter) urls
    text = re.sub(r"http://t.co/[a-zA-Z0-9čČšŠžŽ]+", "", text)
    text = re.sub(r"https://t.co/[a-zA-Z0-9čČšŠžŽ]+", "", text)

    # remove all hashtags or @name Mentions (Usernames only allowed to includes characters A-Z, 0-9 and underscores)
    text = re.sub(r"[@#][a-zA-Z0-9_čČšŠžŽ]+", "", text)

    # remove non alphabetical characters
    text = re.sub(r"[^a-zA-Z0-9\sčČšŠžŽ]", "", text)

    # remove multiple white spaces
    text = re.sub(' +', ' ', text)

    # convert all letters to lower case
    text = text.lower()

    return text.strip()

