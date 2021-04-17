import pandas as pd, os
from typing import Union, List


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
