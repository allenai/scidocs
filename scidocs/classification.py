import json
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from lightning.classification import LinearSVC


np.random.seed(1)


def classify(X_train, y_train, X_test, y_test):
    """
    Simple classification methods using sklearn framework.
    Selection of C happens inside of X_train, y_train via
    cross-validation. 
    
    Arguments:
        X_train, y_train -- training data
        X_test, y_test -- test data to evaluate on (can also be validation data)

    Returns: 
        F1 on X_test, y_test (out of 100), rounded to two decimal places
    """
    estimator = LinearSVC(loss="squared_hinge", random_state=42)
    Cs = np.logspace(-4, 2, 7)
    svm = GridSearchCV(estimator=estimator, cv=3, param_grid={'C': Cs})
    svm.fit(X_train, y_train)
    preds = svm.predict(X_test)
    return 100 * f1_score(y_test, preds, average='macro')


def get_X_y_for_classification(embeddings, train_path, val_path, test_path):
    """
    Given the directory with train/test/val files for mesh classification
        and embeddings, return data as X, y pair
        
    Arguments:
        embeddings: embedding dict
        mesh_dir: directory where the mesh ids/labels are stored
        dim: dimensionality of embeddings

    Returns:
        X, y: dictionaries of training X and training y
              with keys: 'train', 'val', 'test'
    """
    dim = len(next(iter(embeddings.values())))
    train = pd.read_csv(train_path)
    val = pd.read_csv(val_path)
    test = pd.read_csv(test_path)
    X = defaultdict(list)
    y = defaultdict(list)
    for dataset_name, dataset in zip(['train', 'val', 'test'], [train, val, test]):
        for s2id, class_label in dataset.values:
            if s2id not in embeddings:
                X[dataset_name].append(np.zeros(dim))
            else:
                X[dataset_name].append(embeddings[s2id])
            y[dataset_name].append(class_label)
        X[dataset_name] = np.array(X[dataset_name])
        y[dataset_name] = np.array(y[dataset_name])
    return X, y