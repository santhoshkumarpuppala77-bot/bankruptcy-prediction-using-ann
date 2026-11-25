# Bankruptcy Prediction Using Artificial Neural Networks (ANN)

## 📌 Project Overview

Bankruptcy is a major financial challenge that can affect the stability and survival of companies. Identifying financially distressed companies at an early stage is important for investors, banks, creditors, and financial institutions to minimize potential financial losses.

This project focuses on developing an **Artificial Neural Network (ANN)** model to predict whether a company is financially **Alive or Failed** based on its historical financial information.

The project also explores class imbalance handling, data preprocessing, neural network optimization, and model evaluation to improve bankruptcy prediction performance.

## 🎯 Business Problem

Companies may experience financial difficulties due to increasing debt, declining profitability, poor cash flow, and other financial challenges.

The objective is to develop a machine learning-based classification system that analyzes company financial attributes and identifies companies associated with bankruptcy risk.

The predictions can support financial institutions, investors, and other stakeholders in making informed financial decisions.

## 📂 Dataset Information

| Attribute | Description |
|---|---|
| Dataset | American Bankruptcy Dataset |
| Domain | Finance and Financial Risk Analysis |
| Total Records | 78,682 |
| Time Period | 1999–2018 |
| Input Features | 20 |
| Numerical Features | 18 |
| Categorical Features | 2 |
| Target Variable | `status_label` |
| Problem Type | Binary Classification |

### Target Variable

The target variable contains two classes:

- `0` – Alive
- `1` – Failed

## ⚙️ Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- TensorFlow / Keras
- Optuna
- Streamlit
- Joblib

## 🔍 Data Preprocessing

The following preprocessing techniques were implemented:

1. **Unnecessary Column Removal:** Removed `company_name` from the input features.
2. **Feature Selection:** Selected 18 numerical and 2 categorical features.
3. **Outlier Handling:** Investigated outliers and their potential impact on model performance.
4. **Feature Scaling:** Applied RobustScaler to numerical features.
5. **Categorical Encoding:** Applied OneHotEncoder to categorical features.
6. **ColumnTransformer:** Combined numerical and categorical preprocessing pipelines.
7. **Target Encoding:** Encoded Alive as 0 and Failed as 1.
8. **Class Imbalance Handling:** Used class weighting and SMOTE experiments to address the minority class.

The fitted preprocessor was saved and reused during prediction.

## 🧠 Model Development

The problem is treated as a **Binary Classification Problem**.

An Artificial Neural Network (ANN) was developed using TensorFlow and Keras.

### ANN Architecture

- Input Layer
- Four Hidden Dense Layers
- Batch Normalization
- Dropout Regularization
- ReLU and Tanh Activation Functions
- Sigmoid Output Layer

### Model Configuration

| Parameter | Value |
|---|---|
| Hidden Layers | 4 |
| Neurons | 128, 32, 128, 64 |
| Optimizer | Adam |
| Learning Rate | 0.0006622442381609686 |
| Loss Function | Binary Cross-Entropy |
| Batch Normalization | Applied |
| Dropout | 0.3, 0.2, 0.1, 0.2 |
| Output Activation | Sigmoid |

### Hyperparameter Optimization

Optuna was used to explore ANN hyperparameters, including:

- Number of hidden layers
- Number of neurons
- Learning rate
- Batch size
- Optimizer
- Activation functions
- Dropout rates

## 📊 Model Evaluation

The model performance was evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- ROC-AUC
- PR-AUC

Special attention was given to Precision and Recall because the dataset contains an imbalanced target variable.

## 🚀 Streamlit Application

A Streamlit application was developed to provide an interactive interface for bankruptcy prediction.

### Application Features

- Company financial information input.
- ANN-based bankruptcy prediction.
- Alive or Failed classification.
- Failed probability display.
- Interactive user interface.

### Run the Application

```bash
streamlit run app.py
```

## 📁 Project Structure

```text
American-Bankruptcy-Prediction/
│
├── app.py
├── ann_model.keras
├── preprocessor.pkl
├── requirements.txt
├── README.md
│
├── notebooks/
│   └── bankruptcy_prediction.ipynb
│
└── images/
    └── application_screenshot.png
```

*The folder structure is a suggested repository organization. Add only the files and folders that exist in your project.*

## 💻 Installation and Setup

### 1. Clone the Repository

```bash
git clone <your-github-repository-url>
```

### 2. Navigate to the Project Folder

```bash
cd American-Bankruptcy-Prediction
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application

```bash
streamlit run app.py
```

## 🔮 Future Improvements

- Improve minority-class prediction performance.
- Experiment with advanced class imbalance handling techniques.
- Perform additional hyperparameter optimization.
- Evaluate model performance on newer financial data.
- Implement model and data drift monitoring.
- Improve the Streamlit dashboard with additional financial risk insights.

## 🏁 Conclusion

This project demonstrates how Artificial Neural Networks can be applied to financial bankruptcy prediction using historical company financial data. By combining data preprocessing, class imbalance handling, hyperparameter optimization, and model evaluation, the project aims to develop a useful system for identifying companies associated with financial failure.

## 👨‍💻 Author
**P.Santhosh Kumar**
