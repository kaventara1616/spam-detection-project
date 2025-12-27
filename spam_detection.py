# Email Spam Detection using Naive Bayes

import pandas as pd
import nltk
import string

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score

from nltk.corpus import stopwords

# Download stopwords
nltk.download('stopwords')

# ----------------------------------
# Load Dataset
# ----------------------------------
# Change file path if needed
data = pd.read_csv("spam.csv", encoding="latin-1")

# Keep only required columns
data = data[['v1', 'v2']]
data.columns = ['label', 'text']

# Convert labels to numbers
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# ----------------------------------
# Text Preprocessing Function
# ----------------------------------
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

data['text'] = data['text'].apply(preprocess_text)

# ----------------------------------
# Train-Test Split
# ----------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    data['text'],
    data['label'],
    test_size=0.2,
    random_state=42
)

# ----------------------------------
# TF-IDF Vectorization
# ----------------------------------
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# ----------------------------------
# Model Training (Naive Bayes)
# ----------------------------------
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# ----------------------------------
# Prediction
# ----------------------------------
y_pred = model.predict(X_test_tfidf)

# ----------------------------------
# Evaluation
# ----------------------------------
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)

# ----------------------------------
# Test with Custom Email
# ----------------------------------
sample_email =["You are a winner U have been specially selected 2 receive å£1000 cash or a 4* holiday (flights inc) speak to a live operator 2 claim 0871277810810"]
sample_vector = vectorizer.transform(sample_email)
prediction = model.predict(sample_vector)

if prediction[0] == 1:
    print("Prediction: SPAM")
else:
    print("Prediction: HAM (Not Spam)")
