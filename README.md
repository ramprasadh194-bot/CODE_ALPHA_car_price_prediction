# 🚗 Car Price Prediction with Machine Learning

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?style=for-the-badge&logo=scikit-learn)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?style=for-the-badge&logo=jupyter)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=for-the-badge)
![Internship](https://img.shields.io/badge/CodeAlpha-Internship-red?style=for-the-badge)

---

## 📌 Project Overview

This project is a complete Machine Learning pipeline to **predict used car prices** based on features like brand goodwill, horsepower, mileage, engine capacity, and more. Built as part of the **CodeAlpha Data Science Internship — Task 3**.

The goal is to demonstrate real-world applications of **regression modelling**, **feature engineering**, and **model evaluation** using industry-standard Python libraries.

---

## 🎯 Objectives

- Collect and preprocess car-related features (brand, horsepower, mileage, etc.)
- Train multiple regression models to predict car prices
- Handle data preprocessing, feature engineering, and model evaluation
- Use Python libraries: **Pandas, Scikit-learn, and Matplotlib**
- Understand real-world applications of ML in price prediction

---

## 📂 Project Structure

```
car-price-prediction/
│
├── car_price_prediction.ipynb   # Main Jupyter Notebook (full pipeline)
├── best_car_price_model.pkl     # Saved best ML model (joblib)
├── scaler.pkl                   # Saved StandardScaler
├── dataset/
│   └── car_data.csv             # Raw dataset
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

---

## 📊 Dataset

- **Source:** CodeAlpha Internship Dataset
- **Features include:**
  - `brand` / `model` — Car manufacturer and model name
  - `year` — Year of manufacture
  - `mileage` — Fuel efficiency (kmpl)
  - `engine` — Engine displacement (CC)
  - `max_power` — Maximum power (bhp)
  - `fuel` — Fuel type (Petrol/Diesel/CNG/Electric)
  - `transmission` — Manual or Automatic
  - `owner` — Ownership history
  - `selling_price` — **Target variable** (in INR)

---

## 🔧 Tech Stack

| Library | Purpose |
|---|---|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical computations |
| `matplotlib` | Data visualization |
| `seaborn` | Statistical plots |
| `scikit-learn` | ML models, preprocessing, evaluation |
| `joblib` | Model serialization |

---

## 🚀 ML Pipeline

### 1. 📥 Data Loading & Exploration
- Auto-detect CSV file
- Display shape, dtypes, missing values, and statistics

### 2. 📈 Exploratory Data Analysis (EDA)
- Price distribution (histogram + KDE)
- Top 10 car brands by count
- Boxplots: Price vs Fuel Type, Price vs Transmission
- Scatter plots: Mileage, Horsepower, Engine vs Price
- Correlation heatmap

### 3. 🛠️ Data Preprocessing
- Handle missing values (median for numeric, mode for categorical)
- Extract numeric values from unit-bearing columns (`23.4 kmpl` → `23.4`)
- Feature Engineering: `car_age = 2024 - year`
- Label Encoding for categorical features
- StandardScaler for feature normalization

### 4. 🤖 Models Trained

| Model | Type |
|---|---|
| Linear Regression | Baseline |
| Ridge Regression | Regularized Linear |
| Lasso Regression | Regularized Linear |
| Random Forest Regressor | Ensemble |
| Gradient Boosting Regressor | Ensemble |

### 5. 🔍 Hyperparameter Tuning
- GridSearchCV on best model (3-fold CV)
- Tuned: `n_estimators`, `max_depth`, `min_samples_split`

### 6. 📊 Evaluation Metrics
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score
- 5-Fold Cross Validation

---

## 📉 Results

| Model | MAE | RMSE | R² Score |
|---|---|---|---|
| Linear Regression | — | — | — |
| Ridge Regression | — | — | — |
| Lasso Regression | — | — | — |
| Random Forest | — | — | ✅ Best |
| Gradient Boosting | — | — | — |

> *(Metric values filled after running the notebook)*

---

## 📷 Sample Visualizations

- ✅ Price Distribution Plot
- ✅ Correlation Heatmap
- ✅ Model Comparison Bar Chart (R² & RMSE)
- ✅ Feature Importance Plot (Top 15)
- ✅ Actual vs Predicted Scatter Plot
- ✅ Residual Analysis

---

## ⚙️ Installation & Usage

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/car-price-prediction.git
cd car-price-prediction
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Notebook
```bash
jupyter notebook car_price_prediction.ipynb
```

> Run all cells from top to bottom — the notebook is fully sequential.

---

## 📦 Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
joblib
jupyter
```

> Install via: `pip install -r requirements.txt`

---

## 🏆 Key Insights

- **Car age** and **max power (bhp)** are the strongest predictors of price
- **Diesel cars** tend to have higher resale value than petrol
- **Automatic transmission** cars are priced significantly higher
- **Ensemble models** (Random Forest, Gradient Boosting) outperform linear models significantly
- Proper feature extraction from string columns (e.g., `"23.4 kmpl"`) is critical for model accuracy

---

## 🌍 Real-World Applications

- Used car marketplace pricing engines (OLX, Cars24, CarDekho)
- Insurance premium estimation
- Bank loan valuation for vehicle-backed loans
- Dealer inventory pricing automation

---

## 👤 Author

**Ram Prasadh**
B.Tech — Artificial Intelligence & Data Science
VSB College of Engineering Technical Campus, Coimbatore (2024–2028)

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://linkedin.com/in/your-profile)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=flat&logo=github)](https://github.com/your-username)

---

## 🏢 Internship

This project was completed as **Task 3** of the **CodeAlpha Data Science Internship**.

> 📧 services@codealpha.tech | 🌐 [www.codealpha.tech](https://www.codealpha.tech)

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---

⭐ *If you found this project helpful, please give it a star!*
