import spacy
import re

from nltk.stem import PorterStemmer,SnowballStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

nlp = spacy.load("en_core_web_sm", disable=['tagger', 'parser', 'ner'])
stopwords = nlp.Defaults.stop_words



def preprocessing(text,stem=0):
    """
    Applies different steps of preprocessing to a text.
    Preprocessing includes:
    - remove all emoticons
    - remove non-standard lexical tokens (which are not numeric or alphabetical)
    - remove url and @name mentions
    - remove standard stopwords (english stopwords from spacy)
    - convert all letters to lower case
    - perform stemming (PorterStemmer nltk, SnowBallStemmer nltk)

    Arguments
    ----------
    text:                   AnyStr
                            Text which should be converted by preprocessing
    Returns
    -------
    preprocessed_text:      String
                            Text which is converted by preprocessing.
    """
    stemmers = [SnowballStemmer("english"), PorterStemmer()]
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
    text = re.sub(r"http://t.co/[a-zA-Z0-9]+", "", text)


    # remove all hashtags or @name Mentions (Usernames only allowed to includes characters A-Z, 0-9 and underscores)
    text = re.sub(r"[@#][a-zA-Z0-9_]+", "", text)

    # remove non alphabetical characters
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)

    # convert all letters to lower case
    text = text.lower()

    # split text to single words
    words = word_tokenize(text)
    tokens = []


    stemmer = stemmers[stem]
    lemmer = WordNetLemmatizer()

    # remove stopwords and words with length 1
    for word in words:
        # if word not in stopwords:
        if len(word) > 1:
            # apply stemmer to single word
            # word = stemmer.stem(word)
            # word = lemmer.lemmatize(word)
            tokens.append(word)

    # convert tokens back to text
    preprocessed_text = ' '.join([str(element) for element in tokens])
    return preprocessed_text

