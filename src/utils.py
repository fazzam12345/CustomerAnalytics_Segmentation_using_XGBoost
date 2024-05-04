import pandas as pd
import joblib

def save_predictions(predictions_reg, predictions_clf, features_df):
    predictions_df = pd.concat(
        [
            pd.DataFrame(predictions_reg).set_axis(['pred_spend'], axis=1),
            pd.DataFrame(predictions_clf)[[1]].set_axis(['pred_prob'], axis=1),
            features_df.reset_index()
        ],
        axis=1
    )
    predictions_df.to_pickle("artifacts/predictions_df.pkl")
    return predictions_df

def load_predictions():
    return pd.read_pickle('artifacts/predictions_df.pkl')

def save_importance(imp_spend_amount_df, imp_spend_prob_df):
    imp_spend_amount_df.to_pickle("artifacts/imp_spend_amount_df.pkl")
    imp_spend_prob_df.to_pickle("artifacts/imp_spend_prob_df.pkl")

def load_importance():
    imp_spend_amount_df = pd.read_pickle("artifacts/imp_spend_amount_df.pkl")
    imp_spend_prob_df = pd.read_pickle("artifacts/imp_spend_prob_df.pkl")
    return imp_spend_amount_df, imp_spend_prob_df

def save_models(xgb_reg_model, xgb_clf_model):
    joblib.dump(xgb_reg_model, 'artifacts/xgb_reg_model.pkl')
    joblib.dump(xgb_clf_model, 'artifacts/xgb_clf_model.pkl')

def load_model(model_type):
    if model_type == 'regression':
        model = joblib.load('artifacts/xgb_reg_model.pkl')
    elif model_type == 'classification':
        model = joblib.load('artifacts/xgb_clf_model.pkl')
    else:
        raise ValueError("Invalid model type. Choose 'regression' or 'classification'.")
    return model
