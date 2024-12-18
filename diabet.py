
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
file_path = 'diabetes.csv'
data = pd.read_csv(file_path)

# Data Analysis
# Basic data analysis: statistical summary
print('Statistical Summary:')
print(data.describe())

# Check for missing values
print('\nMissing Values:')
print(data.isnull().sum())

# Visualization
# Plot histograms for all features
plt.figure(figsize=(16, 10))
data.hist(bins=20, figsize=(16, 12), color='blue', edgecolor='black')
plt.tight_layout()
plt.savefig('histograms.png')  # Save histograms as an image
plt.show()

# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Correlation Heatmap")
plt.savefig('correlation_heatmap.png')  # Save heatmap as an image
plt.show()

# Machine Learning: Decision Tree Model
# Define features (X) and the target (y)
X = data.drop(columns='Outcome')  # Features
y = data['Outcome']  # Target (Outcome)

# Split the data into training and test sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the DecisionTreeClassifier model
model = DecisionTreeClassifier(random_state=42)

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")