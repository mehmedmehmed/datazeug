import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

df_train = pd.read_csv("customer_churn_train_cleaned.csv")
df_test = pd.read_csv("customer_churn_test_cleaned.csv")

ohe = OneHotEncoder(sparse_output=False)

g = df_train["Gender"]
gen = df_train["Gender"].array.reshape(-1, 1)

columns = pd.unique(g)

features = columns.tolist()

gender = ohe.fit_transform(gen)

df_gender = pd.DataFrame(gender, columns=ohe.categories_)

print(df_gender)

