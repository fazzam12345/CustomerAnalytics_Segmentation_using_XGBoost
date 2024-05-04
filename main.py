from src.data_preparation import load_and_clean_data, time_splitting, feature_engineering
from src.cohort_analysis import cohort_analysis
from src.utils import save_predictions, save_importance, save_models, load_predictions, load_model, load_importance
from src.machine_learning import machine_learning
from src.predictions import get_high_spend_probability_customers, get_recent_purchasers_unlikely_to_buy, get_missed_big_spender_opportunities

def main():
    # Load and clean data
    file_path = "data\\CDNOW_master.txt" # Update this path if necessary
    df = load_and_clean_data(file_path)

    # Perform cohort analysis
    cohort_analysis(df)

    # Split data into temporal in and out sets
    n_days = 90
    temporal_in_df, temporal_out_df = time_splitting(df, n_days)

    # Engineer features
    features_df = feature_engineering(temporal_in_df, temporal_out_df)

    # Train machine learning models and get predictions
    predictions_reg, predictions_clf, imp_spend_amount_df, imp_spend_prob_df, xgb_reg_model, xgb_clf_model = machine_learning(features_df)

    # Save predictions, model importance and trained models
    save_predictions(predictions_reg, predictions_clf, features_df)
    save_importance(imp_spend_amount_df, imp_spend_prob_df)
    save_models(xgb_reg_model, 'regression')
    save_models(xgb_clf_model, 'classification')
        
    # Load models, predictions and model importance
    xgb_reg_model, xgb_clf_model = load_model('regression'), load_model('classification')
    predictions_df = load_predictions()
    imp_spend_amount_df, imp_spend_prob_df = load_importance()

    # Get, print high spend probability customers, recent purchasers unlikely to buy and missed big spender opportunities
    print(get_high_spend_probability_customers(predictions_df))
    print(get_recent_purchasers_unlikely_to_buy(predictions_df))
    print(get_missed_big_spender_opportunities(predictions_df))
    
# Run the main function
if __name__ == "__main__":
    main()