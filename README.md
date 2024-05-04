Customer Lifetime Value Prediction Project - README
Project Overview

This project predicts customer lifetime value (CLTV) and analyzes customer behavior for a music store using historical purchase data. We leverage machine learning and data science techniques to extract valuable insights and support data-driven decisions.

Project Structure

data: Stores the raw data file (CDNOW_master.txt).
src: Contains Python scripts for data processing, analysis, and modeling.
data_preparation.py: Handles data loading, cleaning, time splitting, and feature engineering.
cohort_analysis.py: Performs cohort analysis to understand customer behavior over time.
machine_learning.py: Trains and evaluates machine learning models for CLTV prediction.
predictions.py: Defines functions to identify high-value customers, churn risks, and missed opportunities.
utils.py: Provides utility functions for saving/loading models, predictions, and feature importance data.
artifacts: Stores trained models, feature importance data, and prediction results.
app.py: Creates a Streamlit application for interactive data exploration and visualization.
app_plot.py: Provides sample code for creating visualizations using Plotly.
environment.yml: Specifies the Python environment and required packages.
requirements.txt: Lists the required Python packages.
init.py: Indicates that the src directory is a Python package.
README.md: This file provides an overview of the project.
Data Preparation and Feature Engineering

Loading and Cleaning:
Import and clean raw data (CDNOW_master.txt).
Parse dates and handle missing values.
Time Splitting:
Split data into:
Temporal In-Sample Data (Training): Purchases before a specified cutoff date.
Temporal Out-of-Sample Data (Prediction/Evaluation): Purchases after the cutoff date.
Feature Engineering:
Target Variables:
spend_90_total: Total spending per customer in the 90 days after the cutoff date.
spend_90_flag: Binary flag indicating if a customer made any purchase in those 90 days.
Predictive Features:
Recency: Days since last purchase.
Frequency: Number of purchases before the cutoff date.
Monetary Value:
price_sum: Total amount spent before the cutoff date.
price_mean: Average transaction value.
Feature Matrix: Combine engineered features and target variables for model training.
Machine Learning and Model Training

Model Selection: Utilize XGBoost for regression and classification:
XGBoost Regression: Predicts the total amount a customer will spend in the next 90 days.
XGBoost Classification: Predicts the probability of a customer making a purchase in the next 90 days.
Hyperparameter Tuning: Employ grid search with cross-validation to find the optimal hyperparameters (e.g., learning rate) using R² for regression and ROC-AUC for classification.
Model Training and Evaluation: Train models with the best hyperparameters and evaluate their performance.
Generating Insights and Predictions

Feature Importance: Analyze the importance of each feature in both models.
Customer Segmentation:
High-value Customers: High predicted spending and/or high purchase probability.
Churn Risks: Low predicted spending and/or low purchase probability, especially recent customers unlikely to return.
Missed Opportunities: Customers with no recent purchases but high predicted spending potential.
Streamlit App: Explore and visualize customer segments interactively based on predicted spending and purchase likelihood.
Project Benefits

This project provides valuable insights into customer behavior and CLTV, enabling data-driven decisions for:

Targeted Marketing Campaigns: Focus on high-value customers and those with high purchase probability.
Customer Retention Strategies: Implement strategies to reduce churn and retain valuable customers.
Resource Allocation: Optimize resource allocation based on customer segmentation and predicted CLTV.
By combining machine learning, data analysis, and interactive visualization, this project provides a comprehensive framework for understanding and predicting customer behavior, ultimately leading to improved business outcomes for the music store.
