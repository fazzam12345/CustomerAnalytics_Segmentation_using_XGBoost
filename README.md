# Customer Analytics Dashboard

The objective of this project is to develop a machine learning model capable of forecasting customer expenditure and the likelihood of purchases within a 90-day assessment timeframe, alongside offering guidance on business analytics. The code is a polished version of that developed by matt dancho https://www.business-science.io/img/business-science-logo.png.
The model is trained using XGBoost, a powerful gradient boosting library, and feature engineering is applied to create relevant features from the raw data.

## Data

The dataset used in this project is `CDNOW_master.txt`, which contains customer purchase data with the following columns:

- `customer_id`: A unique identifier for each customer.
- `date`: The date of the purchase.
- `quantity`: The quantity of items purchased.
- `price`: The price of the items purchased.

## Data Preparation

The `load_and_clean_data` function in `data_preparation.py` reads the raw data from the `CDNOW_master.txt` file and performs data cleaning steps, including:

1. Assigning proper column names.
2. Converting the `date` column to a datetime format.
3. Removing any rows with missing values.

## Time Splitting

The `time_splitting` function in `data_preparation.py` splits the data into two sets:

1. **Temporal In-Set**: This set contains customer purchase data up to a specified cutoff date (90 days before the maximum date in the dataset).
2. **Temporal Out-Set**: This set contains customer purchase data after the cutoff date, representing the 90-day evaluation period.

## Feature Engineering

The `feature_engineering` function in `data_preparation.py` creates the following features from the temporal in-set data:

1. **Recency**: The number of days since the customer's last purchase before the cutoff date.
2. **Frequency**: The number of purchases made by the customer before the cutoff date.
3. **Monetary Value**: The total amount spent by the customer (`price_sum`) and the average amount spent per purchase (`price_mean`) before the cutoff date.

Additionally, the function calculates the following target variables from the temporal out-set data:

1. **Spend Amount (`spend_90_total`)**: The total amount spent by the customer during the 90-day evaluation period.
2. **Spend Flag (`spend_90_flag`)**: A binary flag indicating whether the customer made a purchase during the 90-day evaluation period.

These features and target variables are combined into a single `features_df` DataFrame, which is used for training the machine learning models.

## Machine Learning Models

The `machine_learning` function in `machine_learning.py` trains two separate XGBoost models:

1. **XGBoost Regressor**: This model predicts the total spend amount (`spend_90_total`) for each customer during the 90-day evaluation period. The objective function used for this model is `"reg:squarederror"`, and grid search cross-validation is performed to tune the `learning_rate` hyperparameter, optimizing for the R-squared metric.

2. **XGBoost Classifier**: This model predicts the probability (`pred_prob`) of a customer making a purchase (`spend_90_flag`) during the 90-day evaluation period. The objective function used for this model is `"binary:logistic"`, and grid search cross-validation is performed to tune the `learning_rate` hyperparameter, optimizing for the ROC-AUC metric.

The function also calculates and stores the feature importance scores for each model using the `gain` importance type.

## Model Deployment

The trained models, along with the predictions and feature importance scores, are saved using the following functions in `utils.py`:

- `save_predictions`: Saves the predictions from both models and the original features DataFrame to a pickle file (`artifacts/predictions_df.pkl`).
- `save_importance`: Saves the feature importance scores for both models to separate pickle files (`artifacts/imp_spend_amount_df.pkl` and `artifacts/imp_spend_prob_df.pkl`).
- `save_models`: Saves the trained XGBoost models to pickle files (`artifacts/xgb_reg_model.pkl` and `artifacts/xgb_clf_model.pkl`).

The `app.py` file contains a Streamlit application that loads the saved models and predictions, allowing users to explore the customer data and visualize the predicted spend amount and probability. Users can filter customers based on the difference between their actual and predicted spend amounts using a slider.

## Getting Started

To run the project locally, follow these steps:

1. Clone the repository.
2. Install the required dependencies by running `pip install -r requirements.txt`.
3. Run the `main.py` script to train the models, generate predictions, and save the artifacts.
4. Launch the Streamlit app by running `streamlit run app.py`.

