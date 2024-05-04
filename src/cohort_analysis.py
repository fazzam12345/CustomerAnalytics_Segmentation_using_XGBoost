import pandas as pd
import plotnine as pn

pn.options.dpi = 300


# 2.0 COHORT ANALYSIS ----
def cohort_analysis(df):
    # - Only the customers that have joined at the specific business day

    # Get Range of Initial Purchases ----
    first_purchase_tbl = df \
        .sort_values(['customer_id', 'date']) \
        .groupby('customer_id') \
        .first()

    print(first_purchase_tbl)

    print(first_purchase_tbl['date'].min())

    print(first_purchase_tbl['date'].max())

    # Visualize: All purchases within cohort

    df \
        .reset_index() \
        .set_index('date') \
        [['price']] \
        .resample(
            rule = "MS"
        ) \
        .sum() \
        .plot()

    # Visualize: Individual Customer Purchases

    ids = df['customer_id'].unique()
    ids_selected = ids[0:10]

    cust_id_subset_df = df \
        [df['customer_id'].isin(ids_selected)] \
        .groupby(['customer_id', 'date']) \
        .sum() \
        .reset_index()

    pn.ggplot(
        pn.aes('date', 'price', group = 'customer_id'),
        data = cust_id_subset_df
    ) \
        + pn.geom_line() \
        + pn.geom_point() \
        + pn.facet_wrap('customer_id') \
        + pn.scale_x_date(
            date_breaks = "1 year",
            date_labels = "%Y"
        )
