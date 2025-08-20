# Credit Card Default Prediction

This project aims to predict credit card defaults based on a dataset containing a variety of demographic, credit, and financial information for credit card clients.

## Project Description

The dataset used in this project contains information on default payments, demographic factors, credit data, history of payment, and bill statements of credit card clients. The goal is to build and evaluate several machine learning models to accurately predict `credit_card_default`, the target variable.

## Data

The dataset (`amex_credit_card.csv`) includes the following columns:

* `customer_id`
* `name`
* `age`
* `gender`
* `owns_car`
* `owns_house`
* `no_of_children`
* `net_yearly_income`
* `no_of_days_employed`
* `occupation_type`
* `total_family_members`
* `migrant_worker`
* `yearly_debt_payments`
* `credit_limit`
* `credit_limit_used(%)`
* `credit_score`
* `prev_defaults`
* `default_in_last_6months`
* `credit_card_default` (Target Variable)

The initial dataset has 45,528 entries and 19 columns.

---

## Data Cleaning and Feature Engineering

The notebook follows a structured approach for data preparation:

* **Handling Missing Values and Outliers**: Rows with a small number of missing values were dropped, specifically for columns such as `migrant_worker`, `yearly_debt_payments`, `total_family_members`, and `credit_score`. Missing values in the `owns_car` column were imputed with the mode, `N` (No). A KNN Imputer was used to fill in missing `no_of_children` values. Outliers were handled in the numerical columns `net_yearly_income` and `credit_limit`.
* **Log Transformation**: To reduce skewness, a logarithmic transformation was applied to `net_yearly_income`, `no_of_days_employed`, and `credit_limit`.
* **Categorical Encoding**: The `gender` column was cleaned by removing an 'XNA' entry. The `occupation_type` column was ordinal encoded based on the default rates for each occupation category.
* **New Feature Creation**: A new feature, `years_experience`, was derived from `no_of_days_employed`.

---

## Modeling

The project uses a pipeline to preprocess the data and train several classification models to predict `credit_card_default`. Due to the imbalanced nature of the target variable (only 8.12% of the dataset corresponds to defaults), the SMOTE-NC technique from the `imblearn` library was applied to balance the dataset before training the models. The following models were trained and evaluated:

* **Logistic Regression**
* **Decision Tree Classifier**
* **Random Forest Classifier**

---

## Results

The performance of the models was evaluated using a variety of metrics, including:

* Accuracy Score
* Recall Score
* Precision Score
* F1 Score
* ROC-AUC Score
* Confusion Matrix

The final notebook provides a detailed comparison of the models' performance, including a visualization of the confusion matrix and the ROC curve for the best-performing model.
