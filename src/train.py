import os
import pandas as pd
import numpy as np

from sklearn import tree, metrics
import joblib
import argparse

import config
import model_dispatcher


def run(fold, model, train_path, output_path):
    df = pd.read_csv(train_path)

    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    x_train = df_train.drop(["target", "kfold"], axis=1)
    y_train = df_train.target

    x_valid = df_valid.drop(["target", "kfold"], axis=1)
    y_valid = df_valid.target

    clf = model_dispatcher.models[model]

    clf.fit(x_train, y_train)

    preds = clf.predict(x_valid)

    accuracy = metrics.accuracy_score(y_valid, preds)
    roc_auc = metrics.roc_auc_score(y_valid, preds)

    print(f"Fold={fold}, Accuracy={accuracy}, ROC_AUC={roc_auc}")

    joblib.dump(clf, os.path.join(output_path, f"dt_{fold}.bin"))

    return accuracy, roc_auc

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--model", type=str, default=config.DEFAULT_MODEL)
    parser.add_argument("--train_path", type=str, default=config.TRAINING_PATH)
    parser.add_argument("--output_path", type=str, default=config.MODEL_OUTPUT)

    args = parser.parse_args()

    model = args.model
    train_path = args.train_path
    output_path = args.output_path

    accuracy = list()
    roc_auc = list()
    if args.fold is None:
        for fold in range(config.N_KFOLDS):
            acc, roc = run(fold, model, train_path, output_path)
            accuracy.append(acc)
            roc_auc.append(roc)
        print(f"Average accruacy: {np.mean(np.asarray(accuracy))}")
        print(f"Average Roc Auc: {np.mean(np.asarray(roc_auc))}")
    else:
        acc, roc = run(args.fold, model, train_path, output_path)