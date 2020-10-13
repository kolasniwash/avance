import os
import pandas as pd

from sklearn import tree, metrics
import joblib

import config


def run(fold):
    df = pd.read_csv(config.TRAINING_PATH)

    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    x_train = df_train.drop(["target", "kfold"], axis=1)
    y_train = df_train.target

    x_valid = df_valid.drop(["target", "kfold"], axis=1)
    y_valid = df_valid.target

    clf = tree.DecisionTreeClassifier()

    clf.fit(x_train, y_train)

    preds = clf.predict(x_valid)

    accuracy = metrics.accuracy_score(y_valid, preds)
    roc_auc = metrics.roc_auc_score(y_valid, preds)

    print(f"Fold={fold}, Accuracy={accuracy}, ROC_AUC={roc_auc}")

    joblib.dump(clf, os.path.join(config.MODEL_OUTPUT, f"dt_{fold}.bin"))

if __name__ == "__main__":

    for fold in range(config.N_KFOLDS):
        run(fold)