import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import streamlit as st

# Set Seaborn style for better aesthetics
sns.set(style="whitegrid")

# Load the Data
base_path = "D:/Usecase_DemadForecasting1/"

# Load each file into a pandas DataFrame
customer_demographics = pd.read_csv(base_path + "CustomerDemographics.csv")
product_info = pd.read_csv(base_path + "ProductInfo.csv")
transactional_data_1 = pd.read_csv(base_path + "Modified_Transactional_data_retail_01.csv")
transactional_data_2 = pd.read_csv(base_path + "Transactional_data_retail_02.csv")

# Clean Data
transactional_data_1.dropna(subset=['StockCode', 'Quantity', 'Price'], inplace=True)
transactional_data_2.dropna(subset=['StockCode', 'Quantity', 'Price'], inplace=True)

# Merge Transactional Data with Product Info
merged_data_1 = pd.merge(transactional_data_1, product_info, on='StockCode', how='inner')
merged_data_2 = pd.merge(transactional_data_2, product_info, on='StockCode', how='inner')

# Concatenate both merged datasets
combined_data = pd.concat([merged_data_1, merged_data_2], ignore_index=True)

# Convert 'InvoiceDate' to datetime
combined_data['InvoiceDate'] = pd.to_datetime(combined_data['InvoiceDate'], format='%d-%m-%Y', dayfirst=True)

# Add a new column 'Revenue'
combined_data['Revenue'] = combined_data['Quantity'] * combined_data['Price']

# Analyze Top 10 Stock Codes by Quantity
grouped_data = combined_data.groupby('StockCode').agg(
    Total_Quantity_Sold=('Quantity', 'sum'),
    Total_Revenue=('Revenue', 'sum')
).reset_index()

# Sort and get top 10 products
top_10_quantity = grouped_data.sort_values(by='Total_Quantity_Sold', ascending=False).head(10)
top_10_stock_codes = top_10_quantity['StockCode'].tolist()

# Streamlit app
st.title("Demand Forecasting")

# Stock code selection
selected_stock_code = st.selectbox("Select a Stock Code", top_10_stock_codes)

# Prepare Data for Selected Stock Code
filtered_data = combined_data[combined_data['StockCode'] == selected_stock_code]
filtered_data.loc[:, 'InvoiceDate'] = pd.to_datetime(filtered_data['InvoiceDate'], format='%d-%m-%Y', dayfirst=True)

# Aggregate data weekly for the selected product
weekly_data = filtered_data.groupby(['StockCode', pd.Grouper(key='InvoiceDate', freq='W')]).agg(
    Total_Quantity=('Quantity', 'sum'),
    Total_Revenue=('Revenue', 'sum')
).reset_index()

# Forecasting Strategy
if not weekly_data.empty:
    product_data = weekly_data[weekly_data['StockCode'] == selected_stock_code]

    # Create features and target
    product_data.set_index('InvoiceDate', inplace=True)
    product_data['Lag_1'] = product_data['Total_Quantity'].shift(1)
    product_data['Lag_2'] = product_data['Total_Quantity'].shift(2)
    product_data.dropna(inplace=True)

    # Features and target variable
    X = product_data[['Lag_1', 'Lag_2']]
    y = product_data['Total_Quantity']

    # Initialize lists to store predictions and errors
    train_predictions = []
    test_predictions = []
    y_train_full = []
    y_test_full = []

    # Time Series Cross-Validation
    for train_index, test_index in TimeSeriesSplit(n_splits=5).split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Decision Tree Regressor
        decision_tree = DecisionTreeRegressor()
        decision_tree.fit(X_train, y_train)
        train_predictions.append(decision_tree.predict(X_train))
        test_predictions.append(decision_tree.predict(X_test))
        y_train_full.append(y_train)
        y_test_full.append(y_test)

    # Combine predictions and actual values for error analysis
    all_train_predictions = np.concatenate(train_predictions)
    all_test_predictions = np.concatenate(test_predictions)
    all_y_train = np.concatenate(y_train_full)
    all_y_test = np.concatenate(y_test_full)

    # Calculate error metrics
    train_rmse = np.sqrt(mean_squared_error(all_y_train, all_train_predictions))
    test_rmse = np.sqrt(mean_squared_error(all_y_test, all_test_predictions))

    # Step 6: Visualizations
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))

    # Demand Overview Plot
    axs[0].plot(product_data.index, product_data['Total_Quantity'], label='Train Actual Demand', color='blue')
    axs[0].scatter(product_data.index[-len(all_test_predictions):], all_test_predictions, label='Test Predicted Demand', color='green')
    axs[0].scatter(product_data.index[-len(all_y_test):], all_y_test, label='Test Actual Demand', color='red')
    axs[0].set_title(f'Actual vs Predicted Demand for {selected_stock_code}')
    axs[0].set_xlabel('Date')
    axs[0].set_ylabel('Demand')
    axs[0].legend()

    # Error Distribution Plot for Training
    sns.histplot(all_train_predictions - all_y_train, bins=30, kde=True, ax=axs[1], color='green')
    axs[1].set_title('Training Error Distribution')
    axs[1].set_xlabel('Error')
    axs[1].set_ylabel('Frequency')

    # Error Distribution Plot for Testing
    sns.histplot(all_test_predictions - all_y_test, bins=30, kde=True, ax=axs[2], color='red')
    axs[2].set_title('Testing Error Distribution')
    axs[2].set_xlabel('Error')
    axs[2].set_ylabel('Frequency')

    plt.tight_layout()
    st.pyplot(fig)

    # Display RMSE
    st.write(f"Training RMSE: {train_rmse:.2f}")
    st.write(f"Testing RMSE: {test_rmse:.2f}")
else:
    st.write("No data available for the selected stock code.")
