import os
import glob
import tkinter as tk
from tkinter import messagebox
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import RandomOverSampler

def balance_dataset(X, y):
    oversampler = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = oversampler.fit_resample(X, y)
    return X_resampled, y_resampled

# Step 1: Load SMS data from files
def load_sms_data(directory):
    sms_data = []
    labels = []

    for file_path in glob.glob(os.path.join(directory, '*.txt')):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            lines = content.split('\n')
            if len(lines) >= 2:
                label = lines[0].lower()
                sms_content = '\n'.join(lines[1:])
                sms_data.append(sms_content)
                labels.append(label)

    return sms_data, labels

# Step 2: Preprocess and vectorize the text data
def preprocess_and_vectorize(sms_data):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
    sms_vectors = vectorizer.fit_transform(sms_data)
    return sms_vectors, vectorizer

# Step 3: Train the SVM model
def train_svm(X_train, y_train):
    svm_model = SVC(kernel='linear', C=1.0, random_state=42)
    svm_model.fit(X_train, y_train)
    return svm_model

# Step 4: Evaluate the model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
    print(classification_report(y_test, y_pred))

# Step 5: Predict SMS
def predict_sms(sms_text, vectorizer, model):
    sms_vector = vectorizer.transform([sms_text])
    prediction = model.predict(sms_vector)
    return prediction[0]

# Main script
if __name__ == "__main__":
    # Specify the directory containing SMS data files
    sms_directory = './sms-data'  # Replace with the actual directory

    # Load SMS data and corresponding labels (spam/ham)
    sms_data, labels = load_sms_data(sms_directory)

    # Split the data into training and testing sets
    X_train_data, X_test_data, y_train, y_test = train_test_split(sms_data, labels, test_size=0.2, random_state=42)

    # Vectorize the SMS data
    X_train, vectorizer = preprocess_and_vectorize(X_train_data)
    X_test = vectorizer.transform(X_test_data)
    
    X_train, y_train = balance_dataset(X_train, y_train)

    # Train the SVM model
    svm_model = train_svm(X_train, y_train)

    # Evaluate the model on the test set
    evaluate_model(svm_model, X_test, y_test)

    # Tkinter GUI
    def on_predict():
        sms_text = sms_entry.get("1.0", tk.END).strip()
        if sms_text:
            prediction = predict_sms(sms_text, vectorizer, svm_model)
            result_label.config(text=f'This SMS is: {prediction.upper()}')
        else:
            messagebox.showwarning("Input Error", "Please enter SMS text to predict.")

    root = tk.Tk()
    root.title("SMS Spam Detector")

    tk.Label(root, text="Enter SMS Text:").pack(pady=10)
    sms_entry = tk.Text(root, height=10, width=50)
    sms_entry.pack(pady=10)

    predict_button = tk.Button(root, text="Predict", command=on_predict)
    predict_button.pack(pady=10)

    result_label = tk.Label(root, text="")
    result_label.pack(pady=10)

    root.mainloop()