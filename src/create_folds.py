import pandas as pd
from sklearn import model_selection


def dataset(folds=5, test_size=0.1):
    df = pd.read_csv("../input/raw_data.csv")

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df.columns = df.columns.str.lower()
    df = df.rename(columns={"status": "target"}) #these lines to be moved to a preprocessing stage.
    df = df.fillna(0)

    print(df.columns)
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
    train_df.to_csv(f"../input/train_folds.csv", index=False)
    test_1.to_csv(f"../input/test_holdout.csv", index=False)
    test_2.to_csv(f"../input/test.csv", index=False)


if __name__ == "__main__":
    dataset()