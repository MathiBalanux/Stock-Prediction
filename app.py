import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Stock Price Prediction",
    page_icon="📈",
    layout="wide"
)


# =====================================================
# TITLE
# =====================================================

st.title("📈 Stock Price Prediction")

st.write(
    "A simple Machine Learning project using "
    "Linear Regression to predict stock prices."
)


# =====================================================
# LOAD DATASET
# =====================================================

try:

    df = pd.read_csv("stock_data.csv")

except FileNotFoundError:

    st.error(
        "stock_data.csv not found. "
        "Please put it in the same folder as app.py."
    )

    st.stop()


# =====================================================
# DISPLAY DATASET
# =====================================================

st.subheader("📊 Stock Dataset")

st.write(
    f"Total records: **{len(df)}**"
)

st.dataframe(
    df.head(10),
    use_container_width=True
)


# =====================================================
# DATA PREPROCESSING
# =====================================================

df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values("Date")

df = df.dropna()


# =====================================================
# FEATURE ENGINEERING
# =====================================================

# Previous day's closing price

df["Previous_Close"] = df["Close"].shift(1)


# Difference between High and Low

df["Daily_Range"] = (
    df["High"] - df["Low"]
)


# Difference between Open and Close

df["Price_Change"] = (
    df["Close"] - df["Open"]
)


# Remove first row because Previous_Close is empty

df = df.dropna()


# =====================================================
# SELECT FEATURES
# =====================================================

features = [
    "Open",
    "High",
    "Low",
    "Volume",
    "Previous_Close",
    "Daily_Range",
    "Price_Change"
]


X = df[features]

y = df["Close"]


# =====================================================
# TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# =====================================================
# CREATE MODEL
# =====================================================

model = LinearRegression()


# =====================================================
# TRAIN MODEL
# =====================================================

model.fit(
    X_train,
    y_train
)


# =====================================================
# PREDICTION
# =====================================================

predictions = model.predict(
    X_test
)


# =====================================================
# MODEL EVALUATION
# =====================================================

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)

r2 = r2_score(
    y_test,
    predictions
)


# =====================================================
# DISPLAY MODEL PERFORMANCE
# =====================================================

st.subheader("🤖 Model Performance")

col1, col2, col3 = st.columns(3)

col1.metric(
    "MAE",
    f"{mae:.2f}"
)

col2.metric(
    "RMSE",
    f"{rmse:.2f}"
)

col3.metric(
    "R² Score",
    f"{r2:.2f}"
)


# =====================================================
# ACTUAL VS PREDICTED
# =====================================================

st.subheader(
    "📈 Actual vs Predicted Prices"
)

result = pd.DataFrame({

    "Actual Price": y_test.values,

    "Predicted Price": predictions

})

st.line_chart(result)


# =====================================================
# STOCK PRICE HISTORY
# =====================================================

st.subheader(
    "📊 Stock Price History"
)

chart_data = df.set_index("Date")

st.line_chart(
    chart_data["Close"]
)


# =====================================================
# NEXT PRICE PREDICTION
# =====================================================

st.subheader(
    "🔮 Next Closing Price Prediction"
)


# Get latest row

latest = df.iloc[-1]


# Prepare input

new_data = pd.DataFrame({

    "Open": [latest["Open"]],

    "High": [latest["High"]],

    "Low": [latest["Low"]],

    "Volume": [latest["Volume"]],

    "Previous_Close": [latest["Previous_Close"]],

    "Daily_Range": [latest["Daily_Range"]],

    "Price_Change": [latest["Price_Change"]]

})


# Predict

next_price = model.predict(
    new_data
)[0]


# =====================================================
# DISPLAY PREDICTION
# =====================================================

col1, col2 = st.columns(2)

col1.metric(
    "Latest Closing Price",
    f"${latest['Close']:.2f}"
)

col2.metric(
    "Predicted Closing Price",
    f"${next_price:.2f}"
)


# =====================================================
# PREDICTION CHANGE
# =====================================================

change = (
    next_price - latest["Close"]
)

percentage = (
    change / latest["Close"]
) * 100


st.write(
    f"Predicted change: **{percentage:.2f}%**"
)


# =====================================================
# DATA SUMMARY
# =====================================================

st.subheader("📋 Dataset Summary")

st.write(
    df.describe()
)


# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.write(
    "Stock Prediction Project | "
    "Machine Learning using Linear Regression"
)