import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

def test(x, y, model):
    y = np.array(y)
    predict = model.predict(x)
    correct = predict == y
    correct_num = correct.sum()
    accuracy = correct_num / y.size
    return int(np.round(accuracy, 2) * 100)

def test_all(xy, get_importances : bool):
    decision_tree = DecisionTreeClassifier(max_depth=4, criterion='entropy')
    decision_tree.fit(xy[0][0], xy[0][1])
    print(f"\t decision_tree test accuracy: {test(xy[1][0], xy[1][1], decision_tree)}%\n")

    xgboost = XGBClassifier(max_depth=4)
    xgboost.fit(xy[0][0], xy[0][1])
    print(f"\t xgboost test accuracy: {test(xy[1][0], xy[1][1], xgboost)}%\n")

    log_regression = LogisticRegression(C=0.4)
    log_regression.fit(xy[0][0], xy[0][1])
    print(f"\t log_regression test accuracy: {test(xy[1][0], xy[1][1], log_regression)}%\n")

    if get_importances:
        importances = decision_tree.feature_importances_
        indices = np.argsort(importances)
        return indices
    
def split_data(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=1)
    xy = [[x_train, y_train], [x_test, y_test]]
    return xy

def test_with_importances(x_prep_num, y, columns, n, scaler):
    if 0 < n and n <= columns.size:
        print(f"TESTING with {n} importances:")
        x_n = x_prep_num[columns[-n:]]
        x_n_scaled = scaler.fit_transform(x_n)
        xy_n = split_data(x_n_scaled, y)
        test_all(xy_n, False)

def main():
    scaler = MinMaxScaler()
    df = pd.read_csv("titanic_prepared.csv", sep=',')
    df = df.drop('useless', axis=1)

    x = df.drop('label', axis=1)
    x_prep_num = x.fillna(x.median())
    x_scaled = scaler.fit_transform(x_prep_num)
    y = df['label']

    print("\nTESTING without importances:")
    xy = split_data(x_scaled, y)
    indices = test_all(xy, True)

    columns = x_prep_num.columns[indices]
    test_with_importances(x_prep_num, y, columns, 2, scaler)
    
main()