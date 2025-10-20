
# 💳 Credit Card Fraud Detection Project

This project aims to build a machine learning model to detect fraudulent credit card transactions. It uses a highly imbalanced dataset and applies an under-sampling technique to train a Logistic Regression model.

---

## 📋 Project Overview

The primary goal is to accurately classify credit card transactions as either legitimate or fraudulent. Due to the nature of the data, where fraudulent cases are extremely rare, this project focuses on a common strategy to handle such class imbalance.

* **Problem:** The dataset is highly imbalanced, with fraudulent transactions (Class 1) representing only a tiny fraction of all transactions.
* **Strategy:** **Under-sampling** is used. We create a new, balanced dataset by taking all fraudulent transactions and an equal number of randomly selected legitimate transactions.
* **Model:** A **Logistic Regression** model is trained on this new balanced dataset.

---

## 📊 Dataset

The dataset contains credit card transactions, with features that are principal components (V1-V28) obtained from PCA, plus `Time` and `Amount`.

* **Total Transactions:** 284,807
* **Legitimate (Class 0):** 284,315
* **Fraudulent (Class 1):** 492

After under-sampling, a new dataset of **984 samples** (492 legit + 492 fraud) is created to train the model.

---

## 🚀 Project Workflow

1.  **Import Dependencies:** Load necessary libraries (`numpy`, `pandas`, `sklearn`).
2.  **Load Data:** Read the `creditcard.csv` file (Note: *The file name is assumed*).
3.  **Data Exploration:**
    * Check the first and last 5 rows (`.head()`, `.tail()`).
    * Get dataset information (`.info()`).
    * Check for missing values (`.isnull().sum()`).
    * Analyze the class distribution (`.value_counts()`).
4.  **Handle Imbalance (Under-Sampling):**
    * Separate the data into legitimate (`Class == 0`) and fraudulent (`Class == 1`) DataFrames.
    * Create a `legit_sample` DataFrame by randomly sampling 492 transactions from the legitimate data.
    * Create a `new_dataset` by concatenating the `legit_sample` and `fraud` DataFrames.
5.  **Prepare Data for Modeling:**
    * Split the `new_dataset` into features (`X`) and the target variable (`Y`).
    * Split the data into training (80%) and testing (20%) sets using `train_test_split`. `stratify=Y` is used to ensure the class distribution is a 50/50 split in both sets.
6.  **Model Training:**
    * Initialize a `LogisticRegression` model.
    * Train the model on the training data (`X_train`, `Y_train`).
7.  **Model Evaluation:**
    * Evaluate the model's performance on both the training and test data using `accuracy_score`.

---

## 📈 Model & Results

A Logistic Regression model was used for this binary classification task.

* **Training Data Accuracy:** 94.54%
* **Test Data Accuracy:** 91.37%

This demonstrates that the model generalizes well to new, unseen data, although the high accuracy is based on the balanced dataset.

---

## 🛠️ Technologies Used

* **Python**
* **NumPy:** For numerical operations.
* **Pandas:** For data manipulation and analysis.
* **Scikit-learn:** For machine learning:
    * `model_selection.train_test_split`
    * `linear_model.LogisticRegression`
    * `metrics.accuracy_score`
* **Jupyter Notebook:** For project development and experimentation.

---

## 🏃 How to Run This Project

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
    cd your-repository-name
    ```
2.  **Install dependencies:**
    ```bash
    pip install numpy pandas scikit-learn
    ```
3.  **Get the data:**
    * Download the dataset (often named `creditcard.csv`) from its source (e.g., Kaggle).
    * Make sure the CSV file is in the same directory as the notebook.
4.  **Run the notebook:**
    * Launch Jupyter Notebook:
      ```bash
      jupyter notebook
      ```
    * Open the `Project 10. Credit Card Fraud Detection.ipynb` file and run the cells.
