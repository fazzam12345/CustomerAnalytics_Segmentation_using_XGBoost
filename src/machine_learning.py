from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import GridSearchCV
import pandas as pd
import plydata.cat_tools as cat 
import plotnine as pn
import matplotlib.pyplot as plt

def machine_learning(features_df):


    X = features_df[['recency', 'frequency', 'price_sum', 'price_mean']]

    # Next 90-Day Spend Prediction ----
    y_spend = features_df['spend_90_total']

    xgb_reg_spec = XGBRegressor(
        objective="reg:squarederror",
        random_state=123
    )

    xgb_reg_model = GridSearchCV(
        estimator=xgb_reg_spec,
        param_grid=dict(
            learning_rate=[0.01, 0.1, 0.3, 0.5]
        ),
        scoring='r2',
        refit=True,
        cv=5
    )

    xgb_reg_model.fit(X, y_spend)
    predictions_reg = xgb_reg_model.predict(X)

    # Next 90-Day Spend Probability ----
    y_prob = features_df['spend_90_flag']

    xgb_clf_spec = XGBClassifier(
        objective="binary:logistic",
        random_state=123
    )

    xgb_clf_model = GridSearchCV(
        estimator=xgb_clf_spec,
        param_grid=dict(
            learning_rate=[0.01, 0.1, 0.3, 0.5]
        ),
        scoring='roc_auc',
        refit=True,
        cv=5
    )

    xgb_clf_model.fit(X, y_prob)
    predictions_clf = xgb_clf_model.predict_proba(X)

    # Feature Importance (Global) ----
    # Importance | Spend Amount Model
    imp_spend_amount_dict = xgb_reg_model \
        .best_estimator_ \
        .get_booster() \
        .get_score(importance_type='gain')

    imp_spend_amount_df = pd.DataFrame(
        data={
            'feature': list(imp_spend_amount_dict.keys()),
            'value': list(imp_spend_amount_dict.values())
        }
    ) \
        .assign(
        feature=lambda x: cat.cat_reorder(x['feature'], x['value'])
    )

    # Importance | Spend Probability Model
    imp_spend_prob_dict = xgb_clf_model \
        .best_estimator_ \
        .get_booster() \
        .get_score(importance_type='gain')

    imp_spend_prob_df = pd.DataFrame(
        data={
            'feature': list(imp_spend_prob_dict.keys()),
            'value': list(imp_spend_prob_dict.values())
        }
    ) \
        .assign(
        feature=lambda x: cat.cat_reorder(x['feature'], x['value'])
    )

    # Visualize and display metrics for the two models
    print("Spend Amount Model Metrics:")
    print(xgb_reg_model.best_score_)
    print(xgb_reg_model.best_params_)

    print("Spend Probability Model Metrics:")
    print(xgb_clf_model.best_score_)
    print(xgb_clf_model.best_params_)
         

    return predictions_reg, predictions_clf, imp_spend_amount_df, imp_spend_prob_df, xgb_reg_model, xgb_clf_model


