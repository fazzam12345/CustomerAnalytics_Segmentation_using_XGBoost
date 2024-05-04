import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pathlib
import base64

# Set Streamlit theme
st.set_page_config(page_title="Customer Analytics Dashboard", layout="wide")

# PATHS
BASE_PATH = pathlib.Path(__file__).parent.resolve()
ART_PATH = BASE_PATH.joinpath("artifacts").resolve()

# DATA
predictions_df = pd.read_pickle(ART_PATH.joinpath("predictions_df.pkl"))

df = predictions_df \
    .assign(
        spend_actual_vs_pred=lambda x: x['spend_90_total'] - x['pred_spend']
    )

# Streamlit App
st.title("Customer Analytics Dashboard")
st.write("Explore Customers by Predicted Spend versus Actual Spend during the 90-day evaluation period.")

# Slider Marks
x = np.linspace(df['spend_actual_vs_pred'].min(), df['spend_actual_vs_pred'].max(), 10, dtype=int)
x = x.round(0)

spend_delta_max = st.slider(
    "Spend Actual vs Predicted",
    min_value=float(df['spend_actual_vs_pred'].min()),
    max_value=float(df['spend_actual_vs_pred'].max()),
    value=float(df['spend_actual_vs_pred'].max()),
    step=100.0, 
    format="%i"
)

st.write("Segment Customers that were predicted to spend but didn't. Then target these customers with targeted emails.")

df_filtered = df[df['spend_actual_vs_pred'] <= spend_delta_max]

# Plot
fig = px.scatter(
    data_frame=df_filtered,
    x='frequency',
    y='pred_prob',
    color='spend_actual_vs_pred',
    color_continuous_midpoint=0,
    opacity=0.5,
    color_continuous_scale='IceFire',
    hover_name='customer_id',
    hover_data=['spend_90_total', 'pred_spend'],
)

fig.update_layout(
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font_color='white',
    height=700
)

fig.update_traces(
    marker=dict(size=12)
)

st.plotly_chart(fig)

fig2 = px.scatter(
    data_frame=df_filtered,
    x='pred_spend',
    y='spend_90_total',
    color='spend_actual_vs_pred',
    color_continuous_midpoint=0,
    opacity=0.5,
    color_continuous_scale='IceFire',
    hover_name='customer_id',
    hover_data=['frequency', 'pred_prob'],
)

fig2.update_layout(
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font_color='white',
    height=700
)

fig2.update_traces(
    marker=dict(size=12)
)

st.plotly_chart(fig2)

# Download Button
csv = df_filtered.to_csv(index=False)
b64 = base64.b64encode(csv.encode()).decode() 
href = f'<a href="data:file/csv;base64,{b64}" download="customer_segmentation.csv">Download Segmentation</a>'
st.markdown(href, unsafe_allow_html=True)
