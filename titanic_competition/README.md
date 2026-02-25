# Titanic Survival Prediction (Machine Learning Project)

This is a self-directed machine learning project based on the Titanic dataset.  
The goal of this project is to predict whether a passenger survived using different machine learning models and feature engineering techniques.

Through this project, I learned how to build a complete machine learning pipeline from raw data to final predictions.

The project includes:

- Data preprocessing
- Feature engineering
- Model training
- Model evaluation
- Hyperparameter tuning
- Model comparison

The models were implemented using **Python, Pandas, and Scikit-Learn**.

---

## Dataset

The dataset is the **Titanic survival dataset**, which contains passenger information such as:

- Passenger class (Pclass)
- Age
- Fare
- Gender
- Embarkation port
- Family relationships

Target variable:

Survived

0 = Did not survive
1 = Survived

---

## Project Workflow

### 1. Data Cleaning and Preprocessing

The raw dataset contained missing values and categorical variables.

Steps performed:

- Dropped irrelevant columns:
  - PassengerId
  - Name
  - Ticket
  - Cabin

- Handled missing values:
  - Age → filled with median
  - Fare → filled with median
  - Embarked → filled with most common value

- Converted categorical features into numerical values using one-hot encoding:
  - Sex
  - Embarked

This helped me understand how real-world datasets must be cleaned before training models.

---

### 2. Feature Engineering

I experimented with different features to improve model performance.

Basic features:

- Pclass
- Age
- Fare
- Sex
- Embarked

Engineered features:

- FamilySize = SibSp + Parch + 1
- IsAlone (whether passenger traveled alone)
- Title extracted from passenger names (Mr, Mrs, Miss, etc.)

Feature engineering significantly improved model performance.

---

### 3. Logistic Regression Model

I first built a Logistic Regression model as a baseline.

Concepts learned:

- Linear classification models
- Feature scaling using StandardScaler
- L1 regularization
- Model coefficients and interpretation
- Train/validation split

Performance:

Validation Accuracy ≈ 0.77
ROC-AUC ≈ 0.83


---

### 4. Model Evaluation

I evaluated models using:

- Accuracy
- Confusion Matrix
- ROC-AUC Score

Example metrics:

Accuracy ≈ 0.77
Confusion Matrix:

[[92 18]
[23 46]]


This helped me understand classification performance beyond accuracy alone.

---

### 5. Random Forest Model

Next, I trained a Random Forest model.

Concepts learned:

- Decision trees
- Ensemble learning
- Bagging
- Feature importance
- Nonlinear relationships

Feature importance showed that the strongest predictors were:

- Title
- Sex
- Fare
- Passenger class
- Age

Performance:

Validation Accuracy ≈ 0.83


Random Forest performed better than Logistic Regression.

---

### 6. Gradient Boosting Model

I also trained a Gradient Boosting model and compared it to Random Forest.

Concepts learned:

- Boosting vs Bagging
- Learning rate effects
- Model comparison

Performance:

Validation Accuracy ≈ 0.81 – 0.83


---

### 7. Hyperparameter Tuning

I used RandomizedSearchCV to improve model performance.

Parameters tuned:

- Number of trees
- Maximum depth
- Minimum samples per split
- Minimum samples per leaf
- Feature selection method

Example best parameters:

n_estimators = 300
max_depth = 5
min_samples_split = 10
min_samples_leaf = 4
max_features = log2


Concepts learned:

- Cross-validation
- Hyperparameter optimization
- Overfitting vs underfitting

---

## Results

Model comparison:

| Model | Validation Accuracy |
|------|----------------------|
| Logistic Regression | ~0.77 |
| Random Forest | ~0.83 |
| Gradient Boosting | ~0.82 |

Random Forest achieved the best performance.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- Jupyter Notebook



