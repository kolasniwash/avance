import pandas as pd
import argparse
import os


import config
"""
Preporcessing is the same across the train and test pipelines.

Capabilities:
Default preprocessing
Feature engineering
Feature selection

Allows different feature creation and selection processed according to the selection of the pipeline.
"""

def default_processing(raw_path):
    """
    Default_processing standardizes the columns

    :return:
    """
    df = pd.read_csv(raw_path)

    df.columns = df.columns.str.lower()
    df = df.rename(columns={"status": "target"})  # these lines to be moved to a preprocessing stage.

    df = df.fillna(0)

    return df



def feature_engineering(df):
    return df

def feature_selection(df):
    return df

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--raw_path", type=str, default=os.path.join(config.RAW_PATH,
                                                                     config.RAW_FILE))
    parser.add_argument("--save_path", type=str, default=os.path.join(config.DATASET_PATH,
                                                                     config.PREPROCESSED_FILE))


    args = parser.parse_args()
    raw_path = args.raw_path
    save_path = args.save_path

    df = default_processing(raw_path)

    df = feature_engineering(df)

    df = feature_selection(df)

    df.to_csv(save_path, index=False)