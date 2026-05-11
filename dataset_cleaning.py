import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
df = pd.read_csv('yt_dataset.csv')

#Getting to know the dataset:
#df.info() #displays each column, number of instances in each column, and data type
#print(df.sample(100).describe()) #provides statistical data on the features of the data set
#print(df.head(5)) #provides the first n lines of the dataset

#checking for duplicates:
print(f"Number of duplicate rows: {sum(df.duplicated())}")
print(f"Total number of rows before: {len(df)}")

df.drop_duplicates(inplace=True)
print(f"Total number of rows before: {len(df)}")

#fixing the Liked column
print(f"\nBefore Liked column edits -->", df["liked"].value_counts(dropna=False))

df["liked"] = df["liked"].replace(
    {
        'yes' : '1',
        'no' : '0',
        '2' : np.nan

    }
)


df["liked"] = pd.to_numeric(df["liked"], errors='coerce')
df.dropna(subset=["liked"], inplace=True)
df["liked"] = df["liked"].astype(int)

print("After Liked column edits -->", df["liked"].value_counts())
print("Rows remaining:", len(df))

#dropping raw time stamps as they do not directly help the model
#dropping id numbers because they carry no predictive meaning on their own
df.drop(["timestamp", "user_id", "video_id"], axis=1, inplace=True)
print("Remaining columns after feature removals:", df.columns.tolist())

#fixing category values
print("Before:", df['category'].value_counts())
category_map = {
    'MUsic':   'Music',
    'music':   'Music',
    'gamingg': 'Gaming',
    'COMEDY':  'Comedy',
    'Ed':      'Education',
    'Tech ':   'Tech'
}

df['category'] = df['category'].str.strip()
df['category'] = df['category'].replace(category_map)

print("\nAfter:", df['category'].value_counts())

#detecting and removing outliers
print("Before:")
print(f"inf values: {np.isinf(df['watch_percent']).sum()}")
print(f"negative values: {(df['watch_percent'] < 0).sum()}")
print(f"values above 1: {(df['watch_percent'] > 1).sum()}")

def filter_outliers(data):
    return data[
        (data['watch_percent'] >= 0) &
        (data['watch_percent'] <= 1) &
        (~np.isinf(data['watch_percent']))
    ]
df = filter_outliers(df)

print(f"\nAfter:")
print(f"watch_percent range: {df['watch_percent'].min():.4f} – {df['watch_percent'].max():.4f}")
print(f"Rows remaining: {len(df)}")

#identifying target feature and seperating from the features
X = df.drop(['subscribed_after'], axis=1)
X = pd.get_dummies(X, columns=['category', 'device', 'watch_time_of_day'])
y = df['subscribed_after']

print("\nFeatures shape:", X.shape)
print("Target distribution:\n", y.value_counts())
print("Subscription rate:", round(y.mean() * 100, 2), "%")

#splitting training data and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=12,
    stratify=y
)

print("Training set: " + str(len(X_train)) + " rows")
print("Test set: " + str(len(X_test)) + " rows")
print("Subscription rate (train): " + str(round(y_train.mean() * 100, 2)) + "%")
print("Subscription rate (test): " + str(round(y_test.mean() * 100, 2)) + "%")

#scaling the training data
scaler = preprocessing.StandardScaler().fit(X_train)
X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

print("Mean of first 3 columns (should be near 0):")
print(X_train_scaled[X_train_scaled.columns[:3]].mean().round(3))
print("Std of first 3 columns (should be near 1):")
print(X_train_scaled[X_train_scaled.columns[:3]].std().round(3))

#create histograms to visualize the data inside the columns after cleaning
numeric_cols = df.select_dtypes(include='number')
numeric_cols.hist(bins=20, figsize=(14, 10), log=True)
plt.tight_layout()
plt.show()

#developing heatmap to identify any problematic correlations between any of the features
plt.figure(figsize=(12, 8))
sns.heatmap(numeric_cols.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Feature correlation heatmap')
plt.tight_layout()
plt.show()

df.to_csv('yt_dataset_cleaned.csv', index=False)