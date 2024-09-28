import joblib
import numpy as np
import pandas as pd
import re
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Load the model and vectorizer
model = joblib.load('logistic_regression_model.joblib')
vectorizer = joblib.load('tfidf_vectorizer.joblib')

# Load your test data (adjust the path)
columns_names = ['target', 'id', 'date', 'flag', 'user', 'text']
twitter_data = pd.read_csv('./mnt/training.1600000.processed.noemoticon.csv', names=columns_names, encoding='ISO-8859-1')

# Preprocess the test data
def stemming(content):
    port_stem = PorterStemmer()
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if word not in stopwords.words('english')]
    return ' '.join(stemmed_content)

# Apply stemming to the text
twitter_data['stemmed_content'] = twitter_data['text'].apply(stemming)

# Prepare the features for the model
X_test = vectorizer.transform(twitter_data['stemmed_content'].values)
y_test = twitter_data['target'].values

# Make predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]  # For ROC curve

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

print('Accuracy Score:', accuracy)
print('Precision Score:', precision)
print('Recall Score:', recall)
print('F1 Score:', f1)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['0', '4'], yticklabels=['0', '4'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.show()

# Classification Report
class_report = classification_report(y_test, y_pred)
print("Classification Report:\n", class_report)

# Precision-Recall Curve
precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_prob)
plt.figure(figsize=(10, 7))
plt.plot(recall_vals, precision_vals, marker='.', label='Logistic Regression')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.savefig('precision_recall_curve.png')
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(10, 7))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('roc_curve.png')
plt.show()

# Class Distribution Bar Plot
plt.figure(figsize=(10, 7))
sns.countplot(x=twitter_data['target'])
plt.title('Class Distribution')
plt.savefig('class_distribution.png')
plt.show()

# Metrics Bar Plot
metrics = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1 Score': f1}
names = list(metrics.keys())
values = list(metrics.values())
plt.figure(figsize=(10, 7))
plt.bar(names, values, color=['blue', 'green', 'red', 'purple'])
plt.title('Metrics Comparison')
plt.savefig('metrics_bar_plot.png')
plt.show()

# Save metrics to a CSV file
metrics_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
    'Score': [accuracy, precision, recall, f1]
})

metrics_df.to_csv('model_metrics.csv', index=False)
print("Model evaluation metrics saved to 'model_metrics.csv'.")
