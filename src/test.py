from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import argparse

import joblib
import pandas as pd
import numpy as np

from sklearn import metrics


import config


def predict(test_path, model, model_dir):

    models = [os.path.join(model_dir, file) for file in os.listdir(model_dir) if model in file]

    print(test_path)
    df = pd.read_csv(test_path)

    X = df.drop("target", axis=1)
    y = df['target']

    scores = list()
    roc_aucs = list()

    for fold, mod in enumerate(sorted(models)):
        clf = joblib.load(mod)

        preds = clf.predict(X)

        accuracy = metrics.accuracy_score(y, preds)
        roc_auc = metrics.roc_auc_score(y, preds)

        scores.append(accuracy)
        roc_aucs.append(roc_auc)

        print(f"Fold={fold}, Accuracy={accuracy}, ROC_AUC={roc_auc}")

    print(f"Average accruacy: {np.mean(np.asarray(scores))}")
    print(f"Average Roc Auc: {np.mean(np.asarray(roc_aucs))}")




if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--test_path", type=str, default=os.path.join(config.DATASET_PATH,
                                                                      config.HOLDOUT_FILE))
    parser.add_argument("--model", type=str, default=config.DEFAULT_MODEL)
    parser.add_argument("--model_dir", type=str, default=config.MODEL_OUTPUT_DIR)


    args = parser.parse_args()

    test_path = args.test_path
    model = args.model
    model_dir = args.model_dir

    predict(test_path, model, model_dir)