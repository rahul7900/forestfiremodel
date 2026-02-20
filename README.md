# 🌲 Forest Fire Prediction – Machine Learning Model

## 📌 Project Overview

Forest fires cause severe environmental, economic, and human losses every year. 
Early prediction of fire-prone conditions can significantly reduce damage through preventive measures and optimized resource allocation.

This project builds a Machine Learning model to predict the likelihood or severity of forest fires using environmental and meteorological data.

---

## 🎯 Objectives

- Perform Exploratory Data Analysis (EDA)
- Identify key environmental factors contributing to forest fires
- Build regression/classification models
- Evaluate model performance using standard ML metrics
- Create a reproducible ML pipeline

---

## 📊 Problem Statement

Given environmental features such as:

- Temperature  
- Relative Humidity  
- Wind Speed  
- Rainfall  
- FFMC (Fine Fuel Moisture Code)  
- DMC (Duff Moisture Code)  
- DC (Drought Code)  
- ISI (Initial Spread Index)  

Predict:

- 🔥 Whether a forest fire will occur (Classification)  
OR  
- 🔥 The burned area (Regression)

---

## 🗂️ Dataset

The dataset used is the UCI Forest Fires Dataset.

### Features

| Feature | Description |
|----------|-------------|
| temp | Temperature (°C) |
| RH | Relative Humidity (%) |
| wind | Wind speed (km/h) |
| rain | Rain (mm/m²) |
| FFMC | Fine Fuel Moisture Code |
| DMC | Duff Moisture Code |
| DC | Drought Code |
| ISI | Initial Spread Index |
| area | Burned area (hectares) |

---

## 🛠️ Tech Stack

- Python 3.x
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost (Optional)
- Jupyter Notebook

---

## 🏗️ Project Structure

forest-fire-prediction/
│
├── data/
│ └── forestfires.csv
│
├── notebooks/
│ └── EDA_and_Model.ipynb
│
├── src/
│ ├── data_preprocessing.py
│ ├── train.py
│ ├── evaluate.py
│ └── utils.py
│
├── models/
│ └── model.pkl
│
├── requirements.txt
└── README.md


---

## 🔍 Exploratory Data Analysis (EDA)

- Distribution analysis of temperature, humidity, wind
- Correlation heatmap
- Feature importance analysis
- Log transformation of burned area (if highly skewed)

---

## 🤖 Model Building

### 1️⃣ Data Preprocessing

- Handling missing values
- Encoding categorical variables
- Feature scaling (StandardScaler / MinMaxScaler)
- Train-Test Split (80-20)

### 2️⃣ Models Implemented

- Linear Regression
- Random Forest Regressor
- Gradient Boosting
- XGBoost (optional)
- Logistic Regression (for classification)

### 3️⃣ Model Evaluation

For Regression:
- R² Score
- MAE
- MSE
- RMSE

For Classification:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

---

## 📈 Sample Results

| Model | RMSE | R² Score |
|--------|------|----------|
| Linear Regression | 2.45 | 0.68 |
| Random Forest | 1.87 | 0.81 |
| XGBoost | 1.72 | 0.85 |

(Random Forest and XGBoost generally perform best.)

---

## 🚀 How to Run the Project

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/forest-fire-prediction.git
cd forest-fire-prediction

2️⃣ Create Virtual Environment
python -m venv venv

Activate environment:

Mac/Linux:
source venv/bin/activate

Windows:
venv\Scripts\activate

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Train the Model
python src/train.py

🔮 Future Improvements

Hyperparameter tuning (GridSearchCV / RandomizedSearchCV)

Cross-validation

Model deployment using Flask / FastAPI

Docker containerization

Streamlit dashboard

CI/CD pipeline integration

MLflow experiment tracking

📚 Key Learnings

Importance of feature engineering

Handling skewed target variables

Ensemble models outperform simple linear models

Real-world environmental data is highly non-linear

📜 License

This project is licensed under the MIT License.

👨‍💻 Author

Rahul Singh
Machine Learning | Data Science | AI

LinkedIn: https://www.linkedin.com/in/rahulsingh792000/


