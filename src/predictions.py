

def get_high_spend_probability_customers(predictions_df):
    return predictions_df.sort_values('pred_prob', ascending=False)

def get_recent_purchasers_unlikely_to_buy(predictions_df):
    return predictions_df[
        (predictions_df['recency'] > -90) &
        (predictions_df['pred_prob'] < 0.20)
    ].sort_values('pred_prob', ascending=False)

def get_missed_big_spender_opportunities(predictions_df):
    return predictions_df[
        predictions_df['spend_90_total'] == 0.0
    ].sort_values('pred_spend', ascending=False)
