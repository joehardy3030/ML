import numpy as np
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler

def iris_data_for_training():
    iris = datasets.load_iris()
    X = iris.data[:, [2,3]]
    y = iris.target
    np.unique(y) # 0 = Iris-Setosa, 1 = Iris-Versicolor, 2 = Iris-Virginica
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)
    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))
    return (X_train, X_test, y_train, y_test, X_train_std, X_test_std, X_combined_std, y_combined)