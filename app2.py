from tkinter import *
import tkinter as tk
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings("ignore")

# Load dataset
dataset = pd.read_csv('diabetes.csv')

# Feature and target variables
feature_cols = ["Glucose", "BloodPressure", "Insulin", "BMI", "Age"]
target_col = "Outcome"
X = dataset[feature_cols]
y = dataset[target_col]

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train classifiers
classifiers = {
    "Logistic Regression": LogisticRegression(),
    "Naive Bayes": BernoulliNB(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Support Vector Machine": SVC(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

for clf_name, clf in classifiers.items():
    clf.fit(X_train_scaled, y_train)

# Create Tkinter GUI
def predict_diabetes(classifier):
    # Get input values
    glucose = float(entry_glucose.get())
    bloodpressure = float(entry_blood.get())
    insulin = float(entry_insulin.get())
    bmi = float(entry_bmi.get())
    age = float(entry_age.get())

    # Scale input values
    input_data = scaler.transform([[glucose,bloodpressure, insulin, bmi, age]])

    # Predict using selected classifier
    prediction = classifier.predict(input_data)
    if prediction[0] == 1:
        result = "Positive for Diabetes.Please consult Doctor"
    else:
        result = "Negative for Diabetes.Take Care"
    
    # Update output label
    lbl_result.config(text=result)

root = tk.Tk()
root.title("Diabetes Prediction")
root.configure(background='black')
root.geometry("600x600+100+100")
root.configure(background="blue")
bg = PhotoImage(file="all.png")
f = ("Cambria", 15)

# Input fields
entry_glucose = tk.Entry(root, width=10, font=("Cambria", 15))  # Increased height using font size
entry_blood = tk.Entry(root, width=10, font=("Cambria", 15))  # Increased height using font size
entry_insulin = tk.Entry(root, width=10, font=("Cambria", 15))  # Increased height using font size
entry_bmi = tk.Entry(root, width=10, font=("Cambria", 15))  # Increased height using font size
entry_age = tk.Entry(root, width=10, font=("Cambria", 15))  # Increased height using font size

# Labels
label_glucose = tk.Label(root, text="Glucose:", bg='blue', fg='white', font=f)
label_blood = tk.Label(root, text="BloodPressure:", bg='blue', fg='white', font=f)
label_insulin = tk.Label(root, text="Insulin:", bg='blue', fg='white', font=f)
label_bmi = tk.Label(root, text="BMI:", bg='blue', fg='white', font=f)
label_age = tk.Label(root, text="Age:", bg='blue', fg='white', font=f)

# Output label
lbl_result = tk.Label(root, text="", bg='blue', fg='white', font=f)

# Place input fields and labels on grid
label_glucose.grid(row=0, column=0, pady=(20, 5))
entry_glucose.grid(row=0, column=1, pady=(20, 5))
label_blood.grid(row=1, column=0, pady=(20, 5))
entry_blood.grid(row=1, column=1, pady=(20, 5))
label_insulin.grid(row=2, column=0, pady=5)
entry_insulin.grid(row=2, column=1, pady=5)
label_bmi.grid(row=3, column=0, pady=5)
entry_bmi.grid(row=3, column=1, pady=5)
label_age.grid(row=4, column=0, pady=5)
entry_age.grid(row=4, column=1, pady=5)

# Output label
lbl_result.grid(row=5, columnspan=2, pady=10)

# Create buttons for each classifier
for i, (clf_name, clf) in enumerate(classifiers.items(), 1):
    btn = tk.Button(root, text=clf_name, command=lambda c=clf: predict_diabetes(c), width=20, bg='white', fg='blue', font=f)
    btn.grid(row=i, column=2, padx=20, pady=10)

root.mainloop()
