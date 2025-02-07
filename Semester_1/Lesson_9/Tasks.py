import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

def draw_importances(importances, indices, features):
    plt.title('Важность признаков')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), features[indices])
    plt.xlabel('Относительная важность')
    plt.show()

def test(x, y, model):
    predict = model.predict(x)
    y = np.array(y)
    correct = predict == y
    correct_num = correct.sum()
    accuracy = correct_num / y.size
    return int(np.round(accuracy, 2) * 100)

def test_all(xy, get_importances : bool):
    print()
    random_forest = RandomForestClassifier(max_depth=4, n_estimators=20, criterion='entropy')
    random_forest.fit(xy[0][0], xy[0][1])
    print(f"\trandom_forest validation accuracy: {test(xy[1][0], xy[1][1], random_forest)}%")
    print(f"\trandom_forest test accuracy: {test(xy[2][0], xy[2][1], random_forest)}%\n")

    xgboost = XGBClassifier(max_depth=4, n_estimators=20)
    xgboost.fit(xy[0][0], xy[0][1])
    print(f"\txgboost validation accuracy: {test(xy[1][0], xy[1][1], xgboost)}%")
    print(f"\txgboost test accuracy: {test(xy[2][0], xy[2][1], xgboost)}%\n")

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(xy[0][0], xy[0][1])
    print(f"\tknn validation accuracy: {test(xy[1][0], xy[1][1], knn)}%")
    print(f"\tknn test accuracy: {test(xy[2][0], xy[2][1], knn)}%\n")

    log_regression = LogisticRegression(C=0.3)
    log_regression.fit(xy[0][0], xy[0][1])
    print(f"\tlog_regression validation accuracy: {test(xy[1][0], xy[1][1], log_regression)}%")
    print(f"\tlog_regression test accuracy: {test(xy[2][0], xy[2][1], log_regression)}%\n")

    if get_importances:
        importances = random_forest.feature_importances_
        indices = np.argsort(importances)
        return importances, indices

def split_data(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=1)
    xy = [[x_train, y_train], [x_val, y_val], [x_test, y_test]]
    return xy

def test_with_importances(x_prep_num, y, columns, n, scaler):
    if 0 < n and n <= columns.size:
        print(f"TESTING with {n} importances:", end='')
        x_n = x_prep_num[columns[-n:]]
        x_n_scaled = scaler.fit_transform(x_n)
        xy_n = split_data(x_n_scaled, y)
        test_all(xy_n, False)

def prepare_num(x):
    x_num = x.drop(['Sex', 'Embarked', 'Pclass'], axis=1)
    x_sex = pd.get_dummies(x['Sex'])
    x_emb = pd.get_dummies(x['Embarked'], prefix='Emb')
    x_pcl = pd.get_dummies(x['Pclass'], prefix='Pclass')

    x_num = pd.concat((x_num, x_sex, x_emb, x_pcl), axis=1)
    return x_num

def get_prep_x_num(df):
    x = df.drop(['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin'], axis=1)
    x_num = prepare_num(x)
    x_prep_num = x_num.fillna(x_num.median())
    return x_prep_num

def main():
    scaler = MinMaxScaler()
    df = pd.read_csv("train.csv", sep=',')

    x_prep_num = get_prep_x_num(df)
    y = df['Survived']

    print("\nTESTING without importances:", end='')
    x_scaled = scaler.fit_transform(x_prep_num)
    xy = split_data(x_scaled, y)
    impotances, indices = test_all(xy, True)

    columns = x_prep_num.columns[indices]
    test_with_importances(x_prep_num, y, columns, 2, scaler)
    test_with_importances(x_prep_num, y, columns, 4, scaler)
    test_with_importances(x_prep_num, y, columns, 8, scaler)

    draw_importances(impotances, indices, x_prep_num.columns)

main()