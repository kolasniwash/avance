import os
import pandas as pd
import argparse
from sklearn import model_selection
import warnings
warnings.filterwarnings('ignore')

import config

def dataset(folds, preprocessed_path, save_directory, test_size=config.DEFAULT_TEST_SIZE):

    df = pd.read_csv(preprocessed_path)

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    test_size = int(df.shape[0] * (1 - test_size))

    # train and test split
    train_df = df[:test_size]
    test_df = df[test_size:]

    # holdout test set
    test_len = test_df.shape[0]//2
    test_1 = test_df[:test_len]
    test_2 = test_df[test_len:]

    # add kfolds to the training set
    train_df['kfold'] = -1

    y = train_df['target'].values

    kf = model_selection.StratifiedKFold(n_splits=folds, shuffle=True, random_state=23)

    for f, (t, v) in enumerate(kf.split(X=train_df, y=y)):
        train_df.loc[v, "kfold"] = f

    # save datasets
    train_df.to_csv(os.path.join(save_directory, config.TRAINING_FILE), index=False)
    test_1.to_csv(os.path.join(save_directory, config.HOLDOUT_FILE), index=False)
    test_2.to_csv(os.path.join(save_directory, config.TEST_FILE), index=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--n_folds", type=int, default=config.N_KFOLDS)
    parser.add_argument("--preprocessed_path", type=str, default=os.path.join(config.DATASET_PATH, config.PREPROCESSED_FILE))
    parser.add_argument("--save_directory", type=str, default=config.DATASET_PATH)

    args = parser.parse_args()

    dataset(args.n_folds,
            args.preprocessed_path,
            args.save_directory)