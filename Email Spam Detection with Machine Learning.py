# ==========================================
# TASK 4: Email Spam Detection using ML
# ==========================================

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ------------------------------------------
# PHASE 1: Load Dataset
# ------------------------------------------

# Load dataset (download from Kaggle and upload to Colab / VS Code)
df = pd.read_csv(r"C:\Users\anisc\Downloads\TASK 4\spam.csv", encoding="latin-1")

# Keep only required columns
df = df[["v1", "v2"]]
df.columns = ["label", "message"]

print(df.head())
print(df.info())

# ------------------------------------------
# Encode Labels (Spam = 1, Ham = 0)
# ------------------------------------------
df["label"] = df["label"].map({"ham": 0, "spam": 1})

# ------------------------------------------
# Text Vectorization (Spam Detector Logic)
# ------------------------------------------
vectorizer = TfidfVectorizer(stop_words="english", max_features=3000)

X = vectorizer.fit_transform(df["message"])
y = df["label"]

# ------------------------------------------
# Train-Test Split
# ------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# PHASE 2: MACHINE LEARNING MODELS
# ==========================================

# ------------------------------------------
# Model 1: Naive Bayes
# ------------------------------------------
nb = MultinomialNB()
nb.fit(X_train, y_train)
nb_pred = nb.predict(X_test)

print("\n--- Naive Bayes Model ---")
print("Accuracy:", accuracy_score(y_test, nb_pred))
print(classification_report(y_test, nb_pred))

# Confusion Matrix
cm_nb = confusion_matrix(y_test, nb_pred)

plt.figure(figsize=(5, 4))
sns.heatmap(cm_nb, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Naive Bayes Confusion Matrix")
plt.show()

# ------------------------------------------
# Model 2: Logistic Regression (Advanced ML)
# ------------------------------------------
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

print("\n--- Logistic Regression Model ---")
print("Accuracy:", accuracy_score(y_test, lr_pred))
print(classification_report(y_test, lr_pred))

# Confusion Matrix
cm_lr = confusion_matrix(y_test, lr_pred)

plt.figure(figsize=(5, 4))
sns.heatmap(cm_lr, annot=True, fmt="d", cmap="Greens")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Logistic Regression Confusion Matrix")
plt.show()

# ------------------------------------------
# Spam Detection Test
# ------------------------------------------
user_message = input("Enter an email: ")

# Transform user input using trained vectorizer
user_message_tfidf = vectorizer.transform([user_message])

# Predict class
prediction = nb.predict(user_message_tfidf)[0]

# Predict probability
probability = nb.predict_proba(user_message_tfidf)[0]

spam_prob = probability[1] * 100
ham_prob = probability[0] * 100

# Output result
print("\n--- Prediction Result ---")
if prediction == 1:
    print("Prediction: SPAM 🚨")
    print(f"Spam Probability: {spam_prob:.2f}%")
else:
    print("Prediction: NOT SPAM ✅")
    print(f"Not Spam Probability: {ham_prob:.2f}%")

feature_names = vectorizer.get_feature_names_out()

# Get log probabilities
spam_log_prob = nb.feature_log_prob_[1]
ham_log_prob = nb.feature_log_prob_[0]

# Difference between spam and ham
word_importance = spam_log_prob - ham_log_prob

# Get words present in the user email
user_words_indices = user_message_tfidf.nonzero()[1]

important_words = sorted(
    [(feature_names[i], word_importance[i]) for i in user_words_indices],
    key=lambda x: x[1],
    reverse=True
)

print("\nReason for Prediction:")
if prediction == 1:
    print("The following words strongly indicate SPAM:")
else:
    print("The following words strongly indicate NOT SPAM:")

for word, score in important_words[:5]:
    print(f"- {word}")