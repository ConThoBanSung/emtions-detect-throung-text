import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import nltk
import joblib  # For saving the model and vectorizer

# Download NLTK stopwords
nltk.download('stopwords')

# Load Twitter data
columns_names = ['target', 'id', 'date', 'flag', 'user', 'text']
twitter_data = pd.read_csv('./mnt/training.1600000.processed.noemoticon.csv', names=columns_names, encoding='ISO-8859-1')

# Data preprocessing
port_stem = PorterStemmer()

def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if word not in stopwords.words('english')]
    return ' '.join(stemmed_content)

# Apply stemming
twitter_data['stemmed_content'] = twitter_data['text'].apply(stemming)
twitter_data.drop(['text'], axis=1, inplace=True)

# Prepare data for training
X = twitter_data['stemmed_content'].values
Y = twitter_data['target'].values
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42)

# Vectorization
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Print accuracy score
print('Accuracy Score:', accuracy_score(y_test, y_pred))

# Save the model and vectorizer
joblib.dump(model, 'logistic_regression_model.joblib')
joblib.dump(vectorizer, 'tfidf_vectorizer.joblib')

print("Model and vectorizer saved successfully.")
