import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# pd.set_option('display.max_columns', None)

df_train = pd.read_csv("customer_churn_train.csv")
df_test = pd.read_csv("customer_churn_test.csv")


def missing_data(x, z):
    """get and show rows with missing data values
       2 arguments x = df1 z = df2"""
    missing_data_train = x[x.isna().any(axis=1)]
    print("Der Trainingsdatensatz enthält %s Zeilen mit fehlenden Werten" % (len(missing_data_train)))
    print(missing_data_train)

    missing_data_test = z[z.isna().any(axis=1)]
    print("Der Testdatensatz enthält %s Zeilen mit fehlenden Werten" % (len(missing_data_test)))
    print(missing_data_test)


numerical_features = ['Age', 'Tenure in Months', 'Tenure in Years', 'Usage Frequency', 'Support Calls',
                      'Payment Delay', 'Last Interaction']

# plt.figure(figsize=(12, 5))
# sns.boxplot(data=pd.melt(df_train[numerical_features]), x='variable', y='value')
# plt.show()


def my_function(y):
    y.drop_duplicates(inplace=True)  # deletes duplicate rows
    y.dropna(inplace=True)  # deletes rows with no data
    y[["Age", "CustomerID"]] = y[["Age", "CustomerID"]].astype(int)  # converts to int
    y["Subscription Type"] = y["Subscription Type"].str.capitalize()
    y["Gender"] = y["Gender"].str.capitalize()
    y.drop(y[y["Age"] > 100].index, inplace=True)  # deletes row if a specific requirement is not met
    y.drop(y[y["Age"] < 0].index, inplace=True)
    y.drop(y[y["Support Calls"] > 80].index, inplace=True)
    y.reset_index()


my_function(df_train)
my_function(df_test)


if __name__ == "__main__":
    def write():
        df_test.to_csv("customer_churn_test_cleaned.csv", index=False)
        df_train.to_csv("customer_churn_train_cleaned.csv", index=False)

    write()

