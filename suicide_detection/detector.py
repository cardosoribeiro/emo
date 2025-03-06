import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv('suicide_detection.csv')

# Feature extraction (TF-IDF)
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)  # Limiting features for efficiency
X = vectorizer.fit_transform(df['text'])

# Label encoding
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['class'])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = MultinomialNB()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)  

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Classification Report
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))   

# Test phrase
test_phrase = 'I want to die!'

# Transform the test phrase using the SAME vectorizer
test_vectorized = vectorizer.transform([test_phrase])

# Make the prediction
prediction = model.predict(test_vectorized)

# Decode the prediction
predicted_class = label_encoder.inverse_transform(prediction)[0]

# Print the result
print(f"The predicted class for the phrase is: {predicted_class}")
