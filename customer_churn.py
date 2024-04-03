import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score
import sklearn.preprocessing as skp
from sklearn.compose import make_column_transformer
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

np.random.seed(seed=42)

pd.set_option('display.max_columns', None)


def train_knn(x_train, yx_train):
    """
    X_train: Data used for training the model
    y_train: Labels used for training the model
    """
    clf = KNeighborsClassifier(n_neighbors=8)
    clf.fit(x_train, yx_train)

    return clf


def evaluate_knn(clf, x, y):
    """
    clf: Trained (knn) classifier
    X: Data to be evaluated
    y: Labels to be evaluated
    """
    # get predictions
    # y_train_pred = cross_val_predict(clf, x, y, cv=5)
    y_train_pred = clf.predict(x)

    # print results
    print("Accuracy: %s" % (accuracy_score(y, y_train_pred)))


df_train = pd.read_csv("customer_churn_train_cleaned.csv")
df_test = pd.read_csv("customer_churn_test_cleaned.csv")

X_train = df_train[df_train.columns.drop(['Churn'])].copy()
y_train = df_train['Churn']

X_test = df_test[df_train.columns.drop(['Churn'])].copy()
y_test = df_test['Churn']

y_train = y_train.replace({'churn': 1, 'no churn': 0})
y_test = y_test.replace({'churn': 1, 'no churn': 0})

sub_order = ["Basic", "Standard", "Premium"]
con_order = ["Monthly", "Quarterly", "Annual"]

enc = skp.OrdinalEncoder(categories=[sub_order, con_order])
ohe = skp.OneHotEncoder(handle_unknown="ignore")

dum_train = pd.get_dummies(X_train["Gender"])
X_train = pd.concat([X_train, dum_train], axis=1)
X_train[["Female", "Male"]] = X_train[["Female", "Male"]].astype(int)

encoded = enc.fit_transform(X_train[["Subscription Type", "Contract Length"]])
encoded_df = pd.DataFrame(encoded, columns=["Subscription Type", "Contract Length"])
df = X_train.drop(X_train[["Gender", "Subscription Type", "Contract Length"]], axis=1)
X_train = pd.concat([df, encoded_df], axis=1)

dum_test = pd.get_dummies(X_test["Gender"])
X_test = pd.concat([X_test, dum_test], axis=1)
X_test[["Female", "Male"]] = X_test[["Female", "Male"]].astype(int)

encoded = enc.fit_transform(X_test[["Subscription Type", "Contract Length"]])
encoded_df = pd.DataFrame(encoded, columns=["Subscription Type", "Contract Length"])
df = X_test.drop(X_test[["Gender", "Subscription Type", "Contract Length"]], axis=1)
X_test = pd.concat([df, encoded_df], axis=1)

X_train = X_train.drop(X_train[["Female", "Subscription Type", "Tenure in Months", "CustomerID", "Tenure in Years",
                                "Usage Frequency"]], axis=1)
X_test = X_test.drop(X_test[["Female", "Subscription Type", "Tenure in Months", "CustomerID", "Tenure in Years",
                             "Usage Frequency"]], axis=1)

X_train = X_train.rename(columns={"Male": "Gender"})
X_test = X_test.rename(columns={"Male": "Gender"})

features = X_train.columns.tolist()

# fig, ax = plt.subplots(figsize=(15, 6))
# sns.heatmap(X_train.corr().round(3), annot=True, fmt=".1g", cmap="coolwarm")
# plt.show()

fig, ax = plt.subplots(figsize=(3, 6))
sns.heatmap(pd.concat([X_train, y_train], axis=1).corr().round(3).iloc[:, -1:], annot=True, fmt='.1g', cmap="Blues_r",
            cbar=False, linewidths=0.5, linecolor='grey')
# plt.tight_layout()
# plt.show()

knn_clf = train_knn(X_train, y_train)
evaluate_knn(knn_clf, X_test, y_test)

X_train_std = X_train.copy()
X_test_std = X_test.copy()

scaler = StandardScaler()

# fit encoder to training data and transform numerical features in training set

X_train_std[features] = scaler.fit_transform(X_train[features])

# transform numerical features in test set
X_test_std[features] = scaler.transform(X_test[features])

knn_clf = train_knn(X_train_std, y_train)
evaluate_knn(knn_clf, X_test_std, y_test)

print(X_train_std)


# als ordinal_encoder: contract length, subscription type
# als dummy = gender, churn
