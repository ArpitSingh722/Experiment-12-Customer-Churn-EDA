"""
Experiment No. 12 (CO-4)
Apply NumPy and Pandas to perform exploratory data analysis (EDA) on a customer
churn dataset by cleaning the data, handling missing values, generating summary
statistics, and identifying significant patterns through visualization.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def perform_eda(csv_filepath: str, chart_output_path: str) -> None:
    if not os.path.exists(csv_filepath):
        raise FileNotFoundError(f"Dataset not found: '{csv_filepath}'")

    # 1. Load dataset
    df = pd.read_csv(csv_filepath)
    print(f"Initial Dataset Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print("-" * 60)

    # 2. Data Cleaning: Remove duplicate records & strip whitespace
    duplicates_count = df.duplicated().sum()
    df = df.drop_duplicates().copy()
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()
    print(f"Duplicate rows removed: {duplicates_count}")

    # Convert TotalCharges to numeric (coercing invalid/blank strings to NaN)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # 3. Handle Missing Values
    print("\nMissing Values Before Imputation:")
    print(df.isnull().sum()[df.isnull().sum() > 0])

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=["object"]).columns

    for col in numeric_cols:
        if df[col].isnull().any():
            median_val = float(np.nanmedian(df[col].to_numpy()))
            df[col] = df[col].fillna(median_val)

    for col in categorical_cols:
        if df[col].isnull().any():
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)

    print(f"Missing Values After Imputation : {df.isnull().sum().sum()}")
    print("-" * 60)

    # 4. Generate Summary Statistics
    print("Numerical Summary Statistics:")
    print(df[numeric_cols].describe().round(2))
    print("-" * 60)

    churn_rate = (df["Churn"] == "Yes").mean() * 100
    print(f"Overall Customer Churn Rate: {churn_rate:.2f}%")
    print("\nChurn Rate by Contract Type (%):")
    contract_churn = pd.crosstab(df["Contract"], df["Churn"], normalize="index") * 100
    print(contract_churn.round(2))
    print("-" * 60)

    # 5. Visualization of Significant Churn Patterns
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    fig.suptitle("Customer Churn Exploratory Data Analysis (EDA)", fontsize=14, fontweight="bold")

    # Plot 1: Overall Churn Distribution
    churn_counts = df["Churn"].value_counts()
    axes[0].bar(churn_counts.index, churn_counts.values, color=["#2b7bba", "#d9534f"], width=0.5)
    axes[0].set_title("Overall Churn Count")
    axes[0].set_xlabel("Churn Status")
    axes[0].set_ylabel("Number of Customers")
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)

    # Plot 2: Churn Rate by Contract Type
    contract_churn["Yes"].plot(kind="bar", ax=axes[1], color="#e67e22", width=0.55)
    axes[1].set_title("Churn Rate (%) by Contract Type")
    axes[1].set_xlabel("Contract Type")
    axes[1].set_ylabel("Churn Percentage (%)")
    axes[1].tick_params(axis="x", rotation=15)
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)

    # Plot 3: Monthly Charges Distribution by Churn
    retained_charges = df[df["Churn"] == "No"]["MonthlyCharges"]
    churned_charges = df[df["Churn"] == "Yes"]["MonthlyCharges"]
    axes[2].boxplot([retained_charges, churned_charges], labels=["No (Retained)", "Yes (Churned)"])
    axes[2].set_title("Monthly Charges vs. Churn")
    axes[2].set_xlabel("Churn Status")
    axes[2].set_ylabel("Monthly Charges ($)")
    axes[2].grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(chart_output_path, dpi=150)
    plt.close(fig)
    print(f"EDA visualization saved successfully to '{chart_output_path}'.")


def main() -> None:
    perform_eda("customer_churn.csv", "churn_eda_summary.png")


if __name__ == "__main__":
    main()
