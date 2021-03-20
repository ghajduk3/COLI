from typing import Tuple, Any

import pandas as pd
from sklearn.model_selection import train_test_split


def balance_data(x:pd.DataFrame, y:pd.DataFrame)->Tuple[pd.DataFrame,pd.DataFrame]:
    """
    Balances data through under-sampling. Selects so many random normal tweets as
    there are hate speech tweets.
    Arguments
    ----------
    x:                      pd.DataFrame
                            The dataframe containing all x values
    y:		                pd.DataFrame
                            The dataframe containing all y values -> hate_speech, shape (n, 1).
    Returns
    ----------
    x_balanced_shuffled:    pd.DataFrame
                            The dataframe containing balanced x values..
    y_balanced_shuffled:	pd.DataFrame
                            The dataframe containing balanced y class label values.
    """

    normal_tweets = y.loc[y == 0]
    hate_speech_tweets = y.loc[y == 1]

    hate_speech_tweets_size = hate_speech_tweets.size

    normal_tweets_random_balanced = normal_tweets.sample(n=hate_speech_tweets_size)

    y_balanced_list = [normal_tweets_random_balanced, hate_speech_tweets]
    y_balanced = pd.concat(y_balanced_list)

    y_balanced_shuffled = y_balanced.sample(frac=1)  # shuffle the whole set

    y_balanced_ind = y_balanced_shuffled.index

    x_balanced_shuffled = x.loc[y_balanced_ind]

    return x_balanced_shuffled, y_balanced_shuffled

def split_dataset(x:pd.DataFrame,y:pd.DataFrame)->Tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    """
    Split input data set into random train and test subsets.
    Arguments
    ----------
    x:                      pd.DataFrame
                            The dataframe containing all x values.
    y:		                pd.DataFrame
                            The dataframe containing all y class labels.
    Returns
    ----------
    X_train:                pd.DataFrame
                            The dataframe containing training X data.

    X_test                  pd.DataFrame
                            The dataframe containing test X data.

    y_train                 pd.DataFrame
                            The dataframe containing all y train class labels.

    y_test                  pd.DataFrame
                            The dataframe containing all y test class labels.
    """
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 42)
    return X_train, X_test, y_train, y_test