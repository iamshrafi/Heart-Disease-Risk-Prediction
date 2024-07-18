import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import messagebox
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle

# Read the dataset
data = pd.read_csv('health_data.csv')

data.plot()

# Define categorical and numerical features
categorical_features = ['Gender']
numerical_features = ['Age', 'Sleep Duration (hours)', 'Blood Pressure (mm Hg)', 
                      'Blood Glucose Level (mg/dL)', 'BMI']

# Preprocessing pipeline with StandardScaler and OneHotEncoder
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ]
)

# Define the logistic regression pipeline
pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('logistic_regression', LogisticRegression())
])

# Split data into training and test sets
X = data.drop(columns=['Health Outcome'])
y = data['Health Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Train the pipeline
pipeline.fit(X_train, y_train)

# Evaluate the model
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Create the Tkinter GUI
window = tk.Tk()
window.title("Heart Disease Risk Prediction")

# Create labels and entry fields for user input
tk.Label(window, text="Age").grid(row=0, column=0)
age_entry = tk.Entry(window)
age_entry.grid(row=0, column=1)

tk.Label(window, text="Sleep Duration (hours)").grid(row=1, column=0)
sleep_duration_entry = tk.Entry(window)
sleep_duration_entry.grid(row=1, column=1)

tk.Label(window, text="Blood Pressure (mm Hg)").grid(row=2, column=0)
blood_pressure_entry = tk.Entry(window)
blood_pressure_entry.grid(row=2, column=1)

tk.Label(window, text="Blood Glucose Level (mg/dL)").grid(row=3, column=0)
blood_glucose_entry = tk.Entry(window)
blood_glucose_entry.grid(row=3, column=1)

tk.Label(window, text="BMI").grid(row=4, column=0)
bmi_entry = tk.Entry(window)
bmi_entry.grid(row=4, column=1)

tk.Label(window, text="Gender").grid(row=5, column=0)
gender_var = tk.StringVar(value="Male")
gender_menu = tk.OptionMenu(window, gender_var, "Male", "Female")
gender_menu.grid(row=5, column=1)

tk.Label(window, text="Family History of Heart Disease").grid(row=6, column=0)
family_history_var = tk.IntVar(value=0)
tk.Radiobutton(window, text="Yes", variable=family_history_var, value=1).grid(row=6, column=1)
tk.Radiobutton(window, text="No", variable=family_history_var, value=0).grid(row=6, column=2)

# Prediction function
def predict_risk():
    try:
        # Collect and validate inputs
        age = float(age_entry.get())
        sleep_duration = float(sleep_duration_entry.get())
        blood_pressure = float(blood_pressure_entry.get())
        blood_glucose = float(blood_glucose_entry.get())
        bmi = float(bmi_entry.get())
        gender = gender_var.get()
        family_history = family_history_var.get() == 1

        # Create a new DataFrame with user inputs
        new_input = pd.DataFrame({
            'Gender': [gender],
            'Age': [age],
            'Sleep Duration (hours)': [sleep_duration],
            'Blood Pressure (mm Hg)': [blood_pressure],
            'Blood Glucose Level (mg/dL)': [blood_glucose],
            'BMI': [bmi],
            'Family History of Heart Disease': [family_history]
        })

        print("New input:", new_input)  # Debug print statement

        # Predict using the pipeline
        prediction = pipeline.predict(new_input)[0]

        # Show the result in a message box
        risk_text = "High risk" if prediction == 1 else "Low risk"
        results = (
            f"Prediction: {risk_text}\n"
            f"Accuracy: {accuracy * 100:.2f}%\n"
            f"Confusion Matrix:\n{conf_matrix}\n"
            f"Classification Report:\n{class_report}"
        )
        
        messagebox.showinfo("Prediction Result", results)

    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numerical values for all fields.")

# Create the prediction button
predict_button = tk.Button(window, text="Predict", command=predict_risk)
predict_button.grid(row=7, column=1)

# Function to open the add data window
def open_add_data_window():
    add_data_window = tk.Toplevel(window)
    add_data_window.title("Add Data")
    
    # Create labels and entry fields for adding data
    tk.Label(add_data_window, text="Gender").grid(row=0, column=0)
    gender_var_add = tk.StringVar(value="Male")
    gender_menu_add = tk.OptionMenu(add_data_window, gender_var_add, "Male", "Female")
    gender_menu_add.grid(row=0, column=1)

    tk.Label(add_data_window, text="Age").grid(row=1, column=0)
    age_entry_add = tk.Entry(add_data_window)
    age_entry_add.grid(row=1, column=1)

    tk.Label(add_data_window, text="Sleep Duration (hours)").grid(row=2, column=0)
    sleep_duration_entry_add = tk.Entry(add_data_window)
    sleep_duration_entry_add.grid(row=2, column=1)

    tk.Label(add_data_window, text="Blood Pressure (mm Hg)").grid(row=3, column=0)
    blood_pressure_entry_add = tk.Entry(add_data_window)
    blood_pressure_entry_add.grid(row=3, column=1)

    tk.Label(add_data_window, text="Blood Glucose Level (mg/dL)").grid(row=4, column=0)
    blood_glucose_entry_add = tk.Entry(add_data_window)
    blood_glucose_entry_add.grid(row=4, column=1)

    tk.Label(add_data_window, text="BMI").grid(row=5, column=0)
    bmi_entry_add = tk.Entry(add_data_window)
    bmi_entry_add.grid(row=5, column=1)

    tk.Label(add_data_window, text="Family History of Heart Disease").grid(row=6, column=0)
    family_history_var_add = tk.IntVar(value=0)
    tk.Radiobutton(add_data_window, text="Yes", variable=family_history_var_add, value=1).grid(row=6, column=1)
    tk.Radiobutton(add_data_window, text="No", variable=family_history_var_add, value=0).grid(row=6, column=2)

    # Function to add the entered data
    def add_data():
        try:
            # Collect and validate inputs
            gender = gender_var_add.get()
            age = float(age_entry_add.get())
            sleep_duration = float(sleep_duration_entry_add.get())
            blood_pressure = float(blood_pressure_entry_add.get())
            blood_glucose = float(blood_glucose_entry_add.get())
            bmi = float(bmi_entry_add.get())
            family_history = family_history_var_add.get() == 1

            # Add data to the dataset
            new_data = pd.DataFrame({
                'Gender': [gender],
                'Age': [age],
                'Sleep Duration (hours)': [sleep_duration],
                'Blood Pressure (mm Hg)': [blood_pressure],
                'Blood Glucose Level (mg/dL)': [blood_glucose],
                'BMI': [bmi],
                'Family History of Heart Disease': [family_history]
            })

            # Append new data to the existing DataFrame and save to CSV
            global data
            data = pd.concat([data, new_data], ignore_index=True)
            data.to_csv('health_data.csv', index=False)

            messagebox.showinfo("Data Added", "New data added successfully.")

        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numerical values for all fields.")

    # Create the update button
    update_button = tk.Button(add_data_window, text="Update", command=add_data)
    update_button.grid(row=7, column=1)

# Create the add data button
add_data_button = tk.Button(window, text="Add Data", command=open_add_data_window)
add_data_button.grid(row=7, column=0)

# now save the model for future use
filename = 'final_model1.nom'
pickle.dump(pipeline, open(filename, 'wb'))

# load the model from disk
loaded_pipeline = pickle.load(open(filename, 'rb'))

# Start the Tkinter event loop
window.mainloop()


