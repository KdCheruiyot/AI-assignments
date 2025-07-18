# Part 2: Practical Implementation (50%)

# --------------------------------------
# Task 1: Classical ML with Scikit-learn
# --------------------------------------

# Importing required libraries
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import pandas as pd
import numpy as np

# Load dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target)

# Handle missing values (for this example, we simulate some missing data)
X.iloc[0, 0] = np.nan
X.fillna(X.mean(), inplace=True)

# Encode labels (already encoded in dataset)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree Classifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Predict and Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='macro'))
print("Recall:", recall_score(y_test, y_pred, average='macro'))


# --------------------------------------
# Task 2: Deep Learning with TensorFlow
# --------------------------------------

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Load MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize and reshape
x_train = x_train / 255.0
x_test = x_test / 255.0
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# One-hot encoding
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Build CNN model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# Evaluate model
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test accuracy:", test_acc)

# Visualize predictions
predictions = model.predict(x_test)
for i in range(5):
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.title(f"Predicted: {np.argmax(predictions[i])}")
    plt.show()


# --------------------------------------
# Task 3: NLP with spaCy
# --------------------------------------

import spacy
nlp = spacy.load("en_core_web_sm")

# Sample text data (mocked Amazon reviews)
reviews = [
    "I love the battery life of this Apple iPhone 13!",
    "Samsung Galaxy S21 is terrible, always overheating!",
    "This Dell laptop works great for school and programming."
]

# Perform NER and sentiment analysis
for review in reviews:
    doc = nlp(review)
    print("\nReview:", review)
    print("Named Entities:")
    for ent in doc.ents:
        print(f" - {ent.text} ({ent.label_})")

    # Rule-based sentiment analysis
    if any(word in review.lower() for word in ["love", "great", "awesome"]):
        sentiment = "Positive"
    elif any(word in review.lower() for word in ["terrible", "bad", "worst"]):
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    print("Sentiment:", sentiment)
