import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt

df_train = pd.read_csv("dogs.csv")
df_test = pd.read_csv("dogs_test.csv")


def entropy(data, column):
    """Berechnet die Entropy von einer Spalte im DataFrame:
    data = DataFrame, column = "Spaltenname" """
    anzahl = len(data)
    info = data.loc[:, column].tolist()
    x = info.count(1) / anzahl
    y = 1 - x
    score = (x * np.log2(x)) + (y * np.log2(y))
    return score * -1


X_train = df_train.drop(["Action"], axis=1)
y_train = df_train["Action"]

columns = df_train.columns


clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)

tree.plot_tree(clf, filled=True, feature_names=columns, proportion=True)
plt.show()



