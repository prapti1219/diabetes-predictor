import tkinter as tk
from tkinter import messagebox
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
df = pd.read_csv('diabetes.csv')
X = df.drop('Outcome', axis=1)
y = df['Outcome']
model = RandomForestClassifier()
model.fit(X, y)
def predict():
    try:
        values = [float(e.get()) for e in entries]
        prediction = model.predict([values])[0]
        result = "Diabetic" if prediction == 1 else "Not Diabetic"
        messagebox.showinfo("Result", f"Prediction: {result}")
    except:
        messagebox.showerror("Error", "Invalid input. Please enter valid numbers.")
root = tk.Tk()
root.title("Diabetes Predictor")

labels = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",
    "BMI", "DiabetesPedigreeFunction", "Age"
]

entries = []

for i, label in enumerate(labels):
    tk.Label(root, text=label).grid(row=i, column=0)
    entry = tk.Entry(root)
    entry.grid(row=i, column=1)
    entries.append(entry)

tk.Button(root, text="Predict", command=predict).grid(row=len(labels), columnspan=2)

root.mainloop()
