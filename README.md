
Customer Analytics Dashboard
This project aims to provide a comprehensive analytics dashboard for customer data, focusing on predicting customer spending behavior and segmenting customers for targeted marketing efforts. The dashboard is built using Streamlit, a popular framework for building interactive web applications for machine learning and data science projects.


Components
The project is structured into several key components, and is a polished version of the code developed by Matt Dancho https://www.business-science.io/img/business-science-logo.png:

1. Streamlit Application (app.py)
The main application is a Streamlit dashboard that visualizes customer spending data. It allows users to explore customers by predicted spend versus actual spend during a 90-day evaluation period. The dashboard includes interactive elements such as sliders to filter data and download segmentation data.

2. Plotly Visualization (app_plot.py)
This script is used for generating scatter plots of customer data, focusing on frequency, predicted probability, and actual vs. predicted spending. It utilizes Plotly for creating interactive and visually appealing plots.

3. Environment Setup (environment.yml)
The environment.yml file is used to create a Conda environment with all the necessary dependencies for running the project. This includes data manipulation libraries (Pandas, NumPy), visualization libraries (Plotly, Matplotlib), and machine learning libraries (XGBoost, Scikit-learn).

4. Data Preparation and Machine Learning (main.py, cohort_analysis.py, data_preparation.py, machine_learning.py, predictions.py, utils.py)
These scripts are responsible for loading and cleaning the data, performing cohort analysis, feature engineering, and training machine learning models to predict customer spending. The models are trained using XGBoost, a powerful gradient boosting framework.

Running the Application
To run the Streamlit application, follow these steps:

Ensure you have Python 3.7.1 installed.
Create a Conda environment using the environment.yml file:
conda env create -f environment.yml
Activate the Conda environment:
conda activate lab_59_cust_lifetime_py
Install the required Python packages listed in requirements.txt:
pip install -r requirements.txt
Run the Streamlit application:
streamlit run app.py
The application will be accessible in your web browser at the URL provided by Streamlit.


