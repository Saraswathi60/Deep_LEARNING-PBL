import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Sample dataset with water quality parameters
data = {
    'pH': [7.0, 6.5, 8.1, 5.0, 7.2, 9.0, 6.8, 8.5, 5.5, 7.8],
    'Turbidity': [1.2, 2.5, 3.0, 4.5, 1.0, 5.0, 2.2, 3.5, 4.8, 2.0],
    'Dissolved_Oxygen': [7.5, 6.8, 5.0, 3.5, 7.2, 4.0, 6.5, 5.8, 3.2, 6.0],
    'Contaminants': [0, 1, 0, 1, 0, 1, 0, 0, 1, 0],  # 0 = Safe, 1 = Unsafe
}

df = pd.DataFrame(data)
df = pd.DataFrame(data)

# Splitting data into training and testing sets
X = df[['pH', 'Turbidity', 'Dissolved_Oxygen']]
y = df['Contaminants']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Decision Tree Classifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Model accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy * 100:.2f}%')

# Function to predict water safety
def predict_water_safety(pH, turbidity, dissolved_oxygen):
    prediction = model.predict([[pH, turbidity, dissolved_oxygen]])
    return "Safe" if prediction[0] == 0 else "Unsafe"

# Example usage
example_pH = 7.0
example_turbidity = 2.0
example_dissolved_oxygen = 6.5
print(f'Water Safety Prediction: {predict_water_safety(example_pH, example_turbidity, example_dissolved_oxygen)}')
