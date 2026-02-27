import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Creating Dataset

data = {
    "text": [
        "Win money now",
        "Limited offer just for you",
        "Hello how are you",
        "Lets meet tomorrow",
        "Claim your free prize",
        "Are we still meeting today",
        "Exclusive deal waiting for you",
        "Please call me when free",
        "Congratulations you won lottery",
        "Project meeting at 5pm",
        "Get cash bonus instantly",
        "Lunch tomorrow?",
        "You have been selected for prize",
        "Can you send the report",
        "Earn money quickly",
        "See you at college",
        "Free entry in contest",
        "Let's complete the assignment",
        "Urgent offer expires soon",
        "Happy birthday!"
    ],
    "label": [
        1,1,0,0,1,0,1,0,1,0,
        1,0,1,0,1,0,1,0,1,0
    ]  # 1 = Spam, 0 = Not Spam
}

df = pd.DataFrame(data)

# Split Data

X = df["text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# Convert Text to Numbers

vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)


# Train Model

model = MultinomialNB()
model.fit(X_train_vectorized, y_train)


# Make Predictions

y_pred = model.predict(X_test_vectorized)


# Evaluate Model

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, zero_division=0))


# Test with New Email

new_email = ["Congratulations! You won a free ticket"]
new_email_vectorized = vectorizer.transform(new_email)
prediction = model.predict(new_email_vectorized)

if prediction[0] == 1:
    print("\nResult: Spam Email")
else:
    print("\nResult: Not Spam Email")
