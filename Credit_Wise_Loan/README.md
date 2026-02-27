
# Credit Wise Loan — Loan Approval Prediction (Binary Classification)

This project builds a **loan approval prediction system** (Approved: Yes/No), making it a **binary classification** problem.  
The workflow is implemented in a Jupyter Notebook and compares multiple machine-learning models to see which performs best.

## Project Structure

- `credit_wise.ipynb` — main notebook (data cleaning → EDA → feature engineering → training & evaluation)
- `loan_approval_data.csv` — dataset used by the notebook
- `README.md` — project documentation (this file)

## Objective

Predict whether a loan will be approved using applicant and loan-related features.  
Models explored in the notebook:
- Logistic Regression
- k-Nearest Neighbors (kNN)
- Naive Bayes (GaussianNB)

## Workflow (What the Notebook Does)

### 1) Load Data
The dataset is loaded from:

```python
df = pd.read_csv("loan_approval_data.csv")
```

### 2) Basic Inspection
The notebook checks:
- first rows (`df.head()`)
- schema and nulls (`df.info()`)
- statistical summary (`df.describe()`)

### 3) Handle Missing Values (Imputation)
Most ML models can’t train with `NaN` values, so the notebook imputes missing data using `SimpleImputer`:

- **Numerical columns** → filled with the **mean**
- **Categorical columns** → filled with the **most frequent** value (mode)

### 4) Exploratory Data Analysis (EDA)
Examples of EDA plots included:
- Class balance pie chart for `Loan_Approved`
- Boxplot of income vs approval
- Histograms (e.g., `Credit_Score` and `Applicant_Income`) split by approval status

### 5) Encode Categorical Features
Because ML models require numeric inputs, categorical fields are encoded:

- Label Encoding is used for:
  - `Education_Level`
  - `Loan_Purpose`

- One-Hot Encoding is applied using `OneHotEncoder(drop="first")` for:
  - `Employment_Status`, `Marital_Status`, `Loan_Purpose`, `Property_Area`,
    `Education_Level`, `Gender`, `Employer_Category`, and also `Loan_Approved`

After encoding, features and target are split:
- **X** = all columns except those containing `"Loan_Approved"`
- **y** = the (encoded) loan approval target column

### 6) Train/Test Split + Scaling
- 80/20 split using `train_test_split(test_size=0.2, random_state=42)`
- Features are standardized using `StandardScaler`

### 7) Train & Evaluate Models
Each model is trained and evaluated on the test set using:
- Precision
- Recall
- F1 Score
- Accuracy
- Confusion Matrix

Models trained:
1. **Logistic Regression**
2. **kNN (n_neighbors=5)**
3. **Gaussian Naive Bayes**

The notebook notes that **kNN performs worse than Logistic Regression** in this experiment.

## Requirements

Typical dependencies used:
- pandas, numpy
- matplotlib, seaborn
- scikit-learn

## How to Run

1. Open the notebook:
   - `Credit_Wise_Loan/credit_wise.ipynb`
2. Ensure `loan_approval_data.csv` is in the same folder as the notebook.
3. Run cells top-to-bottom to reproduce preprocessing, plots, and model evaluation.
