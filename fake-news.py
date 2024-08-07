import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import re
import string

# Load datasets
df_fake = pd.read_csv('Fake.csv')
df_true = pd.read_csv('True.csv')

# Preview datasets
print(df_fake.head())
print(df_true.head())

# Add class labels
df_fake["class"] = 0
df_true["class"] = 1

# Check shapes
print(df_fake.shape, df_true.shape)

# Removing last 10 rows for manual testing
df_fake_manual_testing = df_fake.tail(10)
for i in range(23480, 23470, -1):
    df_fake.drop([i], axis=0, inplace=True)

df_true_manual_testing = df_true.tail(10)
for i in range(21416, 21406, -1):
    df_true.drop([i], axis=0, inplace=True)

# Verify shapes after dropping rows
print(df_fake.shape, df_true.shape)

# Assign class labels for manual testing
df_fake_manual_testing["class"] = 0
df_true_manual_testing["class"] = 1

# Preview manual testing datasets
print(df_fake_manual_testing.head(10))
print(df_true_manual_testing.head(10))

# Example text extraction
txts = df_true_manual_testing["text"]
res = txts.iloc[0]  # Changed from [21407] to [0] as 21407 may not be valid index

# Combine manual testing datasets
df_manual_testing = pd.concat([df_fake_manual_testing, df_true_manual_testing], axis=0)
df_manual_testing.to_csv("manual_testing.csv", index=False)

# Merge datasets
df_merge = pd.concat([df_fake, df_true], axis=0)

# Drop unnecessary columns and reset index
df = df_merge.drop(["title", "subject", "date"], axis=1)
df = df.sample(frac=1).reset_index(drop=True)

# Preview dataset
print(df.head())

# Define text preprocessing function
def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)    
    return text

# Apply text preprocessing
df["text"] = df["text"].apply(wordopt)
x = df["text"]
y = df["class"]

# Split dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

# Vectorize text data
vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

# Train a Decision Tree Classifier
DT = DecisionTreeClassifier()
DT.fit(xv_train, y_train)

# Predict and evaluate the model
pred_dt = DT.predict(xv_test)
print("Model Accuracy:", DT.score(xv_test, y_test))
print(classification_report(y_test, pred_dt))

# Define function to map output labels
def output_label(n):
    return "Fake News" if n == 0 else "Not A Fake News"

# Define manual testing function
def manual_testing(news):
    testing_news = {"text": [news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt)
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)
    pred_DT = DT.predict(new_xv_test)
    return print("\n\nDT Prediction: {}".format(output_label(pred_DT[0])))

# Manual testing example
news = str(input("Enter news text: "))
manual_testing(news)
