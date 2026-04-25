# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Input dataset (hours studied, marks).
2.Train simple linear regression model (find m and b).
3.Give new input (hours studied).
4.Predict marks using Y=mX+b. 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: NARENDRA KRISHNAN K S
RegisterNumber: 212225240096 
*/
# Step 1: Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 2: Create Dataset (Hours studied vs Marks scored)
data = {
    "Hours_Studied": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Marks_Scored":  [35, 40, 50, 55, 60, 65, 70, 80, 85, 95]
}
df = pd.DataFrame(data)

# Display dataset
print("Dataset:\n", df.head())
df
# Step 3: Split into Features and Target
X = df[["Hours_Studied"]]   # Independent variable
y = df["Marks_Scored"]      # Dependent variable

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# Step 5: Train Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)
# Step 6: Predictions
y_pred = model.predict(X_test)

# Step 7: Model Evaluation
print("\nModel Parameters:")
print("Intercept (b0):", model.intercept_)
print("Slope (b1):", model.coef_[0])

print("\nEvaluation Metrics:")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# Step 8: Visualization
plt.figure(figsize=(8,6))
plt.scatter(X, y, color='blue', label="Actual Data")
plt.plot(X, model.predict(X), color='red', linewidth=2, label="Regression Line")
plt.xlabel("Hours Studied")
plt.ylabel("Marks Scored")
plt.title("Simple Linear Regression: Predicting Marks")
plt.legend()
plt.grid(True)
plt.show()
# Step 9: Predict Marks for custom input
hours = 7.5
predicted_marks = model.predict([[hours]])
print(f"\nPredicted marks for {hours} hours of study = {predicted_marks[0]:.2f}")
```

## Output:
<img width="362" height="542" alt="image" src="https://github.com/user-attachments/assets/dc6e732e-ba85-4e8d-a4bf-d10622293678" />
<img width="248" height="94" alt="image" src="https://github.com/user-attachments/assets/7631a8d4-b138-465d-86d0-68bc921f97d7" />
<img width="443" height="168" alt="image" src="https://github.com/user-attachments/assets/75e0b316-fd5f-4cf5-9e29-efd0d0e056bd" />
<img width="742" height="600" alt="image" src="https://github.com/user-attachments/assets/6dfcd895-2ea0-4b36-a753-04075b6fb5ec" />
<img width="759" height="146" alt="image" src="https://github.com/user-attachments/assets/19f52d2a-3b0c-4c1c-8451-b2d260a73cae" />

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
