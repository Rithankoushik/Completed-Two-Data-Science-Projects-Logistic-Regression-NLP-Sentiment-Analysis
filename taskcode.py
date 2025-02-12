import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.linear_model import LogisticRegression # type: ignore
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the dataset
file_path = "student_dataset.csv"  # Update this if needed
df = pd.read_csv(file_path)

# Step 1: Data Exploration

# Check for missing values
print("🔹 Missing Values in Dataset:")
print(df.isnull().sum())

# Identify outliers using boxplots
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.boxplot(y=df["Study Hours"])
plt.title("Boxplot - Study Hours")

plt.subplot(1, 2, 2)
sns.boxplot(y=df["Attendance"])
plt.title("Boxplot - Attendance")

plt.show()

# Plot relationship between Study Hours, Attendance, and Pass
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df["Study Hours"], y=df["Attendance"], hue=df["Pass"], palette="coolwarm", alpha=0.7)
plt.title("Study Hours vs Attendance (Colored by Pass/Fail)")
plt.xlabel("Study Hours")
plt.ylabel("Attendance (%)")
plt.show()

# Step 2: Model Training

# Define features and target variable
X = df[['Study Hours', 'Attendance']]
y = df['Pass']

# Split dataset into train-test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 3: Model Evaluation

# Predictions
y_pred = model.predict(X_test)

# Accuracy score
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {accuracy:.2f}")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("\n🔹 Confusion Matrix:")
print(conf_matrix)

# Confusion Matrix Visualization
plt.figure(figsize=(5, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Fail', 'Pass'], yticklabels=['Fail', 'Pass'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
