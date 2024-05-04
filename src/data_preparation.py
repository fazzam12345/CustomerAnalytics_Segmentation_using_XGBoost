import pandas as pd

def load_and_clean_data(file_path):
    cdnow_raw_df = pd.read_csv(
        file_path,
        sep="\s+",
        names=["customer_id", "date", "quantity", "price"]
    )

    cdnow_df = cdnow_raw_df \
        .assign(
            date=lambda x: x['date'].astype(str)
        ) \
        .assign(
            date=lambda x: pd.to_datetime(x['date'])
        ) \
        .dropna()

    return cdnow_df

def time_splitting(df, n_days):
    max_date = df['date'].max()
    cutoff = max_date - pd.to_timedelta(n_days, unit="d")

    temporal_in_df = df[df['date'] <= cutoff]
    temporal_out_df = df[df['date'] > cutoff]

    return temporal_in_df, temporal_out_df

def feature_engineering(temporal_in_df, temporal_out_df):
    # Make Targets from out data ----
    targets_df = temporal_out_df \
        .drop('quantity', axis=1) \
        .groupby('customer_id') \
        .sum() \
        .rename({'price': 'spend_90_total'}, axis=1) \
        .assign(spend_90_flag=1)

    # Make Recency (Date) Features from in data ----
    max_date = temporal_in_df['date'].max()

    recency_features_df = temporal_in_df \
        [['customer_id', 'date']] \
        .groupby('customer_id') \
        .apply(
        lambda x: (x['date'].max() - max_date) / pd.to_timedelta(1, "day")
    ) \
        .to_frame() \
        .set_axis(["recency"], axis=1)

    # Make Frequency (Count) Features from in data ----
    frequency_features_df = temporal_in_df \
        [['customer_id', 'date']] \
        .groupby('customer_id') \
        .count() \
        .set_axis(['frequency'], axis=1)

    # Make Price (Monetary) Features from in data ----
    price_features_df = temporal_in_df \
        .groupby('customer_id') \
        .aggregate(
        {
            'price': ["sum", "mean"]
        }
    ) \
        .set_axis(['price_sum', 'price_mean'], axis=1)

    # Combine Features ----
    features_df = pd.concat(
        [recency_features_df, frequency_features_df, price_features_df], axis=1
    ) \
        .merge(
        targets_df,
        left_index=True,
        right_index=True,
        how="left"
    ) \
        .fillna(0)

    return features_df