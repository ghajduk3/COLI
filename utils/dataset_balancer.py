from typing import Tuple, Any
from imblearn.over_sampling import RandomOverSampler,SMOTE
from imblearn.under_sampling import RandomUnderSampler

import pandas as pd
from sklearn.model_selection import train_test_split


def balance_data_under_sample(x:pd.DataFrame, y:pd.DataFrame)->Tuple[pd.DataFrame,pd.DataFrame]:
    """
    Balances data through under-sampling.
    Arguments
    ----------
    x                       pd.DataFrame
                            The dataframe containing all x values
    y 		                pd.DataFrame
                            The dataframe containing all y values -> hate_speech, shape (n, 1).
    Returns
    ----------
    x_balanced              pd.DataFrame
                            The dataframe containing balanced x values.
    y_balanced       	    pd.DataFrame
                            The dataframe containing balanced y class label values.
    """


    sampler = RandomOverSampler(random_state=42)
    # sampler = SMOTE()
    x_balanced, y_balanced = sampler.fit_resample(x.to_numpy().reshape(-1,1),y.to_numpy().reshape(-1,1))
    return pd.DataFrame(data=x_balanced.flatten(),columns=['preprocessed']),pd.DataFrame(data=y_balanced.flatten(),columns=['Label'])
def balance_data_over_sample(x:pd.DataFrame, y:pd.DataFrame)->Tuple[pd.DataFrame,pd.DataFrame]:
    """
    Balances data through over-sampling. 
    Arguments
    ----------
    x                       pd.DataFrame
                            The dataframe containing all x values
    y 		                pd.DataFrame
                            The dataframe containing all y values -> hate_speech, shape (n, 1).
    Returns
    ----------
    x_balanced              pd.DataFrame
                            The dataframe containing balanced x values.
    y_balanced       	    pd.DataFrame
                            The dataframe containing balanced y class label values.
    """


    sampler = RandomOverSampler(sampling_strategy=1)
    # sampler = SMOTE()
    x_balanced, y_balanced = sampler.fit_resample(x.to_numpy().reshape(-1,1),y.to_numpy().reshape(-1,1))
    return pd.DataFrame(data=x_balanced.flatten(),columns=['preprocessed']),pd.DataFrame(data=y_balanced.flatten(),columns=['Label'])

def balance_dataset(x:pd.DataFrame,y:pd.DataFrame,method='OVER')->Tuple[pd.DataFrame,pd.DataFrame]:
    """
     Balances input dataset through over, or undersampling
     Arguments
     ----------
     x                   pd.DataFrame
                         Input dataset.

     y                   pd.DataFrame
                         Input class labels.

     method              AnyStr
                         Balance method, currently available :
                         UNDER,OVER or NONE
     Returns
     -------
     data_balanced       Tuple[pd.DataFrame,pd.DataFrame]
                         Balanced text data and its labels.
     """
    if method == 'OVER':
        return balance_data_over_sample(x,y)
    elif method == 'UNDER':
        return balance_data_under_sample(x,y)
    else:
        return x,y


def split_dataset(x:pd.DataFrame,y:pd.DataFrame)->Tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    """
    Split input data set into random train and test subsets.
    Arguments
    ----------
    x                       pd.DataFrame
                            The dataframe containing all x values.
    y		                pd.DataFrame
                            The dataframe containing all y class labels.
    Returns
    ----------
    X_train                 pd.DataFrame
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