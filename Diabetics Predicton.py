# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
file_path = "/content/diabetes - diabetes.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']

# Load dataset skipping the header row
data = pd.read_csv(file_path, skiprows=1, names=names)

# Check for missing values and handle them (if any)
data.dropna(inplace=True)  # This removes rows with missing values, adapt as needed

# Convert all columns to numeric (except the target variable)
data = data.apply(pd.to_numeric, errors='coerce')

# Drop rows with missing values (if any)
data.dropna(inplace=True)

# Split dataset into features and target variable
X = data.drop('class', axis=1)
y = data['class']
# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100)

# Train the classifier
rf_classifier.fit(X_train, y_train)

# Predict the response for test dataset
y_pred = rf_classifier.predict(X_test)

# Model Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Print some of the predicted and actual values
print("Predicted values:", y_pred[:10])
print("Actual values:", y_test[:10].values)
# Print whether the person is diabetic or not based on predicted values
for i in range(10):  # Print for the first 10 samples
    if y_pred[i] == 1:
        print("Person", i+1, "is diabetic.")
    else:
        print("Person", i+1, "is not diabetic.")