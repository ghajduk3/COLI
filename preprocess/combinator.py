import pandas as pd, os
from typing import Union, List

def normalize_eng_dataset_5(dataset_5: pd.DataFrame ):
    """
    Map from dataset 5 class labels into combined dataset class labels
    Archaic : 0  -> Other : 5
    Class: 1 -> Intelligence : 3
    Disability: 2 -> Intelligence : 3
    Ethnicity: 3 -> Racism : 1
    Gender: 4 -> Sexual : 2
    Nationality: 5 -> Racism : 1
    Religion: 6 -> Other : 5
    Sexual Orientation: 7 -> Sexual : 2
    """
    class_transfer_map = {
        0: 5,
        1: 3,
        2: 3,
        3: 1,
        4: 2,
        5: 1,
        6: 5,
        7: 2
    }
    dataset_5['Label'] = dataset_5['Label'].apply(lambda x : class_transfer_map[x])
    return dataset_5

def normalize_eng_dataset_6(dataset_6 : pd.DataFrame):
    """
    Map from dataset 6 class labels into combined dataset class labels
    Appearance: 0 -> Appearance : 4
    Intelligence: 1 -> Intelligence : 3
    Political: 2 -> Other : 5
    Racial: 3 -> Racism : 1
    Sextual: 4 -> Sexual : 2
    """
    class_transfer_map = {
        0: 4,
        1: 3,
        2: 5,
        3: 1,
        4: 2
    }
    dataset_6['Label'] = dataset_6['Label'].apply(lambda x : class_transfer_map[x])
    return dataset_6

def normalize_slo_dataset_2(dataset_2 : pd.DataFrame):
    """
    Map from dataset 2 class labels into combined dataset class labels
    0 Non hate -> Non hate : 0
    1 racism -> Racism : 1
    2 migrants -> Other : 5
    3 islamophobia -> Other : 5
    4 antisemitism -> Racism : 1
    5 religion -> Other : 5
    6 homophobia -> Sexual : 2
    7 sexism -> Sexual : 2
    8 ideology -> Other : 5
    9 media -> Other : 5
    10 politics -> Other : 5
    11 individual -> Other : 5
    12 other -> Other : 5
    """
    class_transfer_map = {
        0: 0,
        1: 1,
        2: 5,
        3: 5,
        4: 1,
        5: 5,
        6: 2,
        7: 2,
        8: 5,
        9: 5,
        10: 5,
        11: 5,
        12: 5
    }
    dataset_2['Label'] = dataset_2['Label'].apply(lambda x : class_transfer_map[x])
    return dataset_2

def read_combine_datasets(data_input_base_path: str, dataset_number=(1, 1), concatenate=True) -> Union[
    pd.DataFrame, List[pd.DataFrame]]:
    """
    Parameters
    ----------
    data_input_base_path
    dataset_number
    concatenate

    Returns
    -------

    """
    all_dataframes = []
    first_ds, last_ds = dataset_number
    for index in range(first_ds, last_ds + 1):
        dataset_path = os.path.join(data_input_base_path, 'dataset_' + str(index), 'data.csv')
        df = pd.read_csv(dataset_path, index_col=None)
        print(dataset_path)
        print(len(df['Label'].to_list()))
        all_dataframes.append(df)
    if concatenate:
        all_dataframes = pd.concat(all_dataframes, axis=0, ignore_index=True)

        return all_dataframes
    else:
        return all_dataframes

