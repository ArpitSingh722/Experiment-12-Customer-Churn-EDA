# Experiment No. 12: Exploratory Data Analysis (EDA) on Customer Churn Dataset

**Course Outcome:** CO-4  
**Student:** Vastav Nandedkar  
**Institution:** School of Computing, MIT Art, Design and Technology University, Pune  

## Aim
Apply NumPy and Pandas to perform exploratory data analysis (EDA) on a customer churn dataset by cleaning the data, handling missing values, generating summary statistics, and identifying significant patterns through visualization.

## Problem Statement
Write a Python program using `numpy`, `pandas`, and `matplotlib` to:
1. Load a customer churn dataset (`customer_churn.csv`).
2. Clean the dataset by removing duplicate records and standardizing data types.
3. Identify and impute missing values (median imputation for numerical features and mode imputation for categorical features).
4. Compute descriptive summary statistics and churn rate breakdowns across contract types and tenure segments.
5. Generate and save multi-panel EDA visualizations (`churn_eda_summary.png`) highlighting key churn patterns.

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/Experiment-12-Customer-Churn-EDA.git
   cd Experiment-12-Customer-Churn-EDA
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the script:
   ```bash
   python customer_churn_eda.py
   ```
4. View the statistical summary in the terminal and inspect the generated chart `churn_eda_summary.png`.
