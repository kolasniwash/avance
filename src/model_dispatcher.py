from sklearn import tree, ensemble, linear_model, svm


models = {
    "tree-gini": tree.DecisionTreeClassifier(criterion="gini",
                                         min_samples_split=20,
                                         min_samples_leaf=20),
    "tree-entropy": tree.DecisionTreeClassifier(criterion="entropy",
                                         min_samples_split=20,
                                         min_samples_leaf=20),
    "rf": ensemble.RandomForestClassifier(n_estimators= 20,
                                           max_depth=18,
                                           min_samples_split=20,
                                           min_samples_leaf=20),
    "logistic": linear_model.LogisticRegression(),
    "svc": svm.SVC(),
    "xgboost": None,
    "lightgbm": None
}


