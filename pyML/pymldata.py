import numpy as np
import pandas as pd
from sklearn import datasets
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
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

def xor_data():
    np.random.seed(0)
    X_xor = np.random.randn(200,2)
    y_xor = np.logical_xor(X_xor[:, 0] > 0, X_xor[:, 1] > 0)
    y_xor = np.where(y_xor, 1, -1) #True gets 1, False gets -1 
    return (X_xor, y_xor)

def wine_data():
    df_wine = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
    df_wine.columns = ['Class label', 'Alcohol',
                       'Malic acid', 'Ash',
                       'Alcalinity of ash', 'Magnesium',
                       'Total phenols', 'Flavinoids',
                       'Nonflavinoid phenols',
                       'Proanthocyanins',
                       'Color intensity', 'Hue',
                       'OD280/OD315 of diluted wines',
                       'Proline']
    print('Class labels', np.unique(df_wine['Class label']))
    return df_wine

def bc_data():
    df_bc =  pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data',
                         header=None)
    return df_bc