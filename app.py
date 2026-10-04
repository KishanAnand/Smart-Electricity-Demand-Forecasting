import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb

from database import (
    init_db,
    ensure_columns,
    log_prediction,
    get_prediction_history,
    get_database_stats
)

st.set_page_config(
    page_title="Smart Energy Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>

.stApp {
    background-color: #ffffff;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

h1 {
    color: #172033 !important;
    font-weight: 700 !important;
    letter-spacing: -0.025em !important;
}

h2 {
    color: #172033 !important;
    font-weight: 650 !important;
}

h3 {
    color: #24324a !important;
    font-weight: 600 !important;
}

section[data-testid="stSidebar"] {
    background-color: #f7f9fc;
    border-right: 1px solid #e5e9f0;
}

div[data-testid="stMetric"] {
    background-color: #f8fafc;
    border: 1px solid #e4e9f1;
    border-radius: 10px;
    padding: 18px 20px;
}

div[data-testid="stMetricLabel"] {
    font-size: 0.88rem;
    font-weight: 500;
    color: #5c667a;
}

div[data-testid="stMetricValue"] {
    font-weight: 650;
    color: #172033;
}

button[data-baseweb="tab"] {
    font-size: 0.95rem;
    font-weight: 500;
    padding-left: 18px;
    padding-right: 18px;
}

button[data-baseweb="tab"][aria-selected="true"] {
    font-weight: 650;
}

div[data-testid="stDataFrame"] {
    border: 1px solid #e5e9f0;
    border-radius: 10px;
    overflow: hidden;
}

div.stButton > button {
    border-radius: 8px;
    font-weight: 600;
    min-height: 42px;
}

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)

POWER_MODEL_FILE = "xgboost_power_model.json"
PEAK_MODEL_FILE = "xgboost_peak_classifier.json"
ANOMALY_MODEL_FILE = "iso_forest_model.pkl"
THRESHOLD_FILE = "energy_thresholds.json"
BENCHMARK_FILE = "model_benchmark.json"
FORECAST_STATE_FILE = "forecast_state.json"
PEAK_METRICS_FILE = "peak_metrics.json"
RESIDUAL_FILE = "residual_std.npy"

FEATURE_COLS = [
    "hour",
    "dayofweek",
    "quarter",
    "month",
    "year",
    "is_weekend",
    "lag_1h",
    "lag_2h",
    "lag_3h",
    "lag_24h",
    "lag_7d",
    "rolling_mean_3h",
    "rolling_mean_6h",
    "rolling_mean_24h",
    "rolling_std_24h"
]

ANOMALY_FEATURES = [
    "lag_1h",
    "lag_2h",
    "lag_3h",
    "lag_24h",
    "lag_7d",
    "rolling_mean_3h",
    "rolling_mean_6h",
    "rolling_mean_24h",
    "rolling_std_24h"
]

init_db()
ensure_columns()

required_files = [
    POWER_MODEL_FILE,
    PEAK_MODEL_FILE,
    ANOMALY_MODEL_FILE,
    THRESHOLD_FILE,
    BENCHMARK_FILE,
    FORECAST_STATE_FILE,
    RESIDUAL_FILE
]

missing_files = [
    file
    for file in required_files
    if not os.path.exists(file)
]

if missing_files:
    st.error("Required project files are missing.")

    for file in missing_files:
        st.write(file)

    st.stop()


@st.cache_resource
def load_models():
    power_model = xgb.XGBRegressor()
    power_model.load_model(POWER_MODEL_FILE)

    peak_model = xgb.XGBClassifier()
    peak_model.load_model(PEAK_MODEL_FILE)

    anomaly_model = joblib.load(ANOMALY_MODEL_FILE)

    return power_model, peak_model, anomaly_model


@st.cache_data
def load_json_file(filename):
    with open(filename, "r") as file:
        return json.load(file)


@st.cache_data
def load_residual_std():
    return float(np.load(RESIDUAL_FILE))


power_model, peak_model, anomaly_model = load_models()

thresholds = load_json_file(THRESHOLD_FILE)
benchmark_data = load_json_file(BENCHMARK_FILE)
forecast_state = load_json_file(FORECAST_STATE_FILE)
residual_std = load_residual_std()

peak_metrics = {}

if os.path.exists(PEAK_METRICS_FILE):
    peak_metrics = load_json_file(PEAK_METRICS_FILE)

DEFAULT_PEAK_PROBABILITY_THRESHOLD = 0.60

try:
    peak_probability_threshold = float(
        peak_metrics["dedicated_classifier"]["probability_threshold"]
    )
except (KeyError, TypeError, ValueError):
    peak_probability_threshold = DEFAULT_PEAK_PROBABILITY_THRESHOLD

model_feature_names = power_model.get_booster().feature_names

if model_feature_names is not None:
    if list(model_feature_names) != FEATURE_COLS:
        st.error(
            "The forecasting model feature structure does not "
            "match the application."
        )

        st.write("Model features:", model_feature_names)
        st.write("Application features:", FEATURE_COLS)

        st.stop()


def recursive_24h_forecast(
    model,
    historical_values,
    start_timestamp
):
    history = [
        float(value)
        for value in historical_values
    ]

    if len(history) < 168:
        raise ValueError(
            "At least 168 historical hourly observations are required."
        )

    start_timestamp = pd.Timestamp(start_timestamp)

    future_timestamps = pd.date_range(
        start=start_timestamp + pd.Timedelta(hours=1),
        periods=24,
        freq="h"
    )

    rows = []

    for timestamp in future_timestamps:
        lag_1h = history[-1]
        lag_2h = history[-2]
        lag_3h = history[-3]
        lag_24h = history[-24]
        lag_7d = history[-168]

        rolling_mean_3h = float(
            np.mean(history[-3:])
        )

        rolling_mean_6h = float(
            np.mean(history[-6:])
        )

        rolling_mean_24h = float(
            np.mean(history[-24:])
        )

        rolling_std_24h = float(
            np.std(
                history[-24:],
                ddof=1
            )
        )

        feature_row = {
            "hour": timestamp.hour,
            "dayofweek": timestamp.dayofweek,
            "quarter": timestamp.quarter,
            "month": timestamp.month,
            "year": timestamp.year,
            "is_weekend": int(timestamp.dayofweek >= 5),
            "lag_1h": lag_1h,
            "lag_2h": lag_2h,
            "lag_3h": lag_3h,
            "lag_24h": lag_24h,
            "lag_7d": lag_7d,
            "rolling_mean_3h": rolling_mean_3h,
            "rolling_mean_6h": rolling_mean_6h,
            "rolling_mean_24h": rolling_mean_24h,
            "rolling_std_24h": rolling_std_24h
        }

        input_df = pd.DataFrame(
            [feature_row]
        )[FEATURE_COLS]

        prediction = float(
            model.predict(input_df)[0]
        )

        prediction = max(
            0.0,
            prediction
        )

        peak_probability = float(
            peak_model.predict_proba(
                input_df
            )[0][1]
        )

        peak_prediction = int(
            peak_probability
            >= peak_probability_threshold
        )

        anomaly_input = input_df[
            ANOMALY_FEATURES
        ]

        anomaly_prediction = int(
            anomaly_model.predict(
                anomaly_input
            )[0]
        )

        lower_bound = max(
            0.0,
            prediction - 1.96 * residual_std
        )

        upper_bound = (
            prediction + 1.96 * residual_std
        )

        rows.append({
            "timestamp": timestamp,
            "predicted_demand_kW": prediction,
            "lower_bound_kW": lower_bound,
            "upper_bound_kW": upper_bound,
            "peak_probability": peak_probability,
            "peak_prediction": peak_prediction,
            "anomaly_prediction": anomaly_prediction
        })

        history.append(prediction)

    return pd.DataFrame(rows)


try:
    forecast_df = recursive_24h_forecast(
        model=power_model,
        historical_values=forecast_state["last_168_values"],
        start_timestamp=forecast_state["last_timestamp"]
    )

except Exception as error:
    st.error(
        "Unable to generate the 24-hour forecast."
    )

    st.exception(error)

    st.stop()


def classify_peak_risk(probability):
    if probability >= 0.80:
        return "Critical"

    if probability >= peak_probability_threshold:
        return "High"

    if probability >= 0.40:
        return "Moderate"

    return "Low"


forecast_df["peak_risk"] = (
    forecast_df["peak_probability"]
    .apply(classify_peak_risk)
)

forecast_df["anomaly_status"] = np.where(
    forecast_df["anomaly_prediction"] == -1,
    "Anomaly",
    "Normal"
)


with st.sidebar:
    st.title("Smart Energy Intelligence")

    st.caption(
        "Machine Learning Analytics Dashboard"
    )

    st.divider()

    st.subheader("System Configuration")

    st.markdown(
        """
**Forecast Model**

XGBoost Regressor

**Peak-Risk Model**

XGBoost Classifier

**Anomaly Model**

Isolation Forest

**Forecast Horizon**

24 Hours

**Engineered Features**

15
"""
    )

    st.divider()

    st.subheader("Peak-Risk Configuration")

    st.metric(
        "Demand Threshold",
        f"{float(thresholds['peak_threshold']):.2f} kW"
    )

    st.metric(
        "Probability Threshold",
        f"{peak_probability_threshold:.0%}"
    )

    st.divider()

    st.subheader("System Status")

    st.success(
        "All models loaded successfully"
    )


st.caption("ENERGY ANALYTICS PLATFORM")

st.title(
    "Smart Electricity Demand Forecasting"
)

st.write(
    "Demand forecasting, peak-risk classification and "
    "anomaly detection using machine learning."
)

st.info(
    "Forecasts are generated from historical household electricity "
    "measurements. The forecast horizon continues from the final "
    "timestamp available in the dataset and does not represent "
    "live utility-grid data."
)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "24-Hour Forecast",
    "Peak Risk",
    "Anomaly Detection",
    "Model Performance",
    "Prediction Logs"
])


with tab1:
    st.caption("FORECASTING")

    st.header(
        "24-Hour Demand Forecast"
    )

    st.caption(
        "Recursive hourly demand predictions generated using "
        "the selected XGBoost regression model."
    )

    average_demand = float(
        forecast_df[
            "predicted_demand_kW"
        ].mean()
    )

    maximum_demand = float(
        forecast_df[
            "predicted_demand_kW"
        ].max()
    )

    maximum_row = forecast_df.loc[
        forecast_df[
            "predicted_demand_kW"
        ].idxmax()
    ]

    peak_hours = int(
        forecast_df[
            "peak_prediction"
        ].sum()
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Average Demand",
        f"{average_demand:.2f} kW"
    )

    col2.metric(
        "Maximum Demand",
        f"{maximum_demand:.2f} kW"
    )

    col3.metric(
        "Peak-Risk Hours",
        peak_hours
    )

    col4.metric(
        "Highest Demand Time",
        pd.Timestamp(
            maximum_row["timestamp"]
        ).strftime(
            "%d %b %H:%M"
        )
    )

    chart_df = forecast_df[
        [
            "timestamp",
            "predicted_demand_kW",
            "lower_bound_kW",
            "upper_bound_kW"
        ]
    ].copy()

    chart_df = chart_df.rename(
        columns={
            "predicted_demand_kW":
                "Predicted Demand",
            "lower_bound_kW":
                "Lower Bound",
            "upper_bound_kW":
                "Upper Bound"
        }
    )

    chart_df = chart_df.set_index(
        "timestamp"
    )

    st.line_chart(
        chart_df
    )

    st.caption(
        "Upper and lower bounds represent an approximate "
        "residual-based 95% prediction interval."
    )

    display_forecast = forecast_df.copy()

    display_forecast["Time"] = (
        pd.to_datetime(
            display_forecast["timestamp"]
        )
        .dt.strftime(
            "%Y-%m-%d %H:%M"
        )
    )

    display_forecast[
        "Predicted Demand (kW)"
    ] = (
        display_forecast[
            "predicted_demand_kW"
        ].round(3)
    )

    display_forecast[
        "Lower Bound (kW)"
    ] = (
        display_forecast[
            "lower_bound_kW"
        ].round(3)
    )

    display_forecast[
        "Upper Bound (kW)"
    ] = (
        display_forecast[
            "upper_bound_kW"
        ].round(3)
    )

    display_forecast[
        "Peak Probability (%)"
    ] = (
        display_forecast[
            "peak_probability"
        ] * 100
    ).round(2)

    display_forecast[
        "Peak Risk"
    ] = (
        display_forecast[
            "peak_risk"
        ]
    )

    st.dataframe(
        display_forecast[
            [
                "Time",
                "Predicted Demand (kW)",
                "Lower Bound (kW)",
                "Upper Bound (kW)",
                "Peak Probability (%)",
                "Peak Risk"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )

    if st.button(
        "Save Forecast to Database",
        type="primary"
    ):
        for _, row in forecast_df.iterrows():
            log_prediction(
                predicted_demand=(
                    row[
                        "predicted_demand_kW"
                    ]
                ),
                forecast_timestamp=(
                    row["timestamp"]
                ),
                lower_bound=(
                    row[
                        "lower_bound_kW"
                    ]
                ),
                upper_bound=(
                    row[
                        "upper_bound_kW"
                    ]
                ),
                peak_probability=(
                    row[
                        "peak_probability"
                    ]
                ),
                peak_prediction=(
                    row[
                        "peak_prediction"
                    ]
                ),
                peak_risk=(
                    row[
                        "peak_risk"
                    ]
                ),
                anomaly_prediction=(
                    row[
                        "anomaly_prediction"
                    ]
                ),
                anomaly_status=(
                    row[
                        "anomaly_status"
                    ]
                )
            )

        st.success(
            "The 24-hour forecast has been saved successfully."
        )


with tab2:
    st.caption(
        "RISK ANALYTICS"
    )

    st.header(
        "Peak Demand Risk Analysis"
    )

    st.caption(
        "Probability-based identification of elevated "
        "electricity demand using the XGBoost classifier."
    )

    max_peak_probability = float(
        forecast_df[
            "peak_probability"
        ].max()
    )

    high_risk_count = int(
        (
            forecast_df[
                "peak_probability"
            ]
            >= peak_probability_threshold
        ).sum()
    )

    peak_threshold_kw = float(
        thresholds[
            "peak_threshold"
        ]
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Peak Demand Threshold",
        f"{peak_threshold_kw:.2f} kW"
    )

    col2.metric(
        "Classifier Threshold",
        f"{peak_probability_threshold:.0%}"
    )

    col3.metric(
        "Highest Peak Probability",
        f"{max_peak_probability:.1%}"
    )

    st.write(
        f"**High-Risk Forecast Hours: "
        f"{high_risk_count} / 24**"
    )

    peak_chart = forecast_df[
        [
            "timestamp",
            "peak_probability"
        ]
    ].copy()

    peak_chart[
        "Peak Probability (%)"
    ] = (
        peak_chart[
            "peak_probability"
        ] * 100
    )

    peak_chart = peak_chart[
        [
            "timestamp",
            "Peak Probability (%)"
        ]
    ]

    peak_chart = peak_chart.set_index(
        "timestamp"
    )

    st.line_chart(
        peak_chart
    )

    high_risk_df = forecast_df[
        forecast_df[
            "peak_prediction"
        ] == 1
    ].copy()

    if high_risk_df.empty:
        st.success(
            "No forecast hour exceeds the selected "
            "peak-risk probability threshold."
        )

    else:
        st.warning(
            f"{len(high_risk_df)} forecast hour(s) exceed "
            "the peak-risk threshold."
        )

        high_risk_display = high_risk_df.copy()

        high_risk_display["Time"] = (
            pd.to_datetime(
                high_risk_display[
                    "timestamp"
                ]
            )
            .dt.strftime(
                "%Y-%m-%d %H:%M"
            )
        )

        high_risk_display[
            "Predicted Demand (kW)"
        ] = (
            high_risk_display[
                "predicted_demand_kW"
            ].round(3)
        )

        high_risk_display[
            "Peak Probability (%)"
        ] = (
            high_risk_display[
                "peak_probability"
            ] * 100
        ).round(2)

        high_risk_display[
            "Peak Risk"
        ] = (
            high_risk_display[
                "peak_risk"
            ]
        )

        st.dataframe(
            high_risk_display[
                [
                    "Time",
                    "Predicted Demand (kW)",
                    "Peak Probability (%)",
                    "Peak Risk"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )

    classifier_metrics = (
        peak_metrics.get(
            "dedicated_classifier",
            {}
        )
    )

    if classifier_metrics:
        st.subheader(
            "Held-Out Classifier Performance"
        )

        m1, m2, m3 = st.columns(3)

        m1.metric(
            "Precision",
            f"{classifier_metrics.get('precision', 0):.3f}"
        )

        m2.metric(
            "Recall",
            f"{classifier_metrics.get('recall', 0):.3f}"
        )

        m3.metric(
            "F1 Score",
            f"{classifier_metrics.get('f1_score', 0):.3f}"
        )

        st.caption(
            "These metrics describe performance on the "
            "historical held-out test period."
        )


with tab3:
    st.caption(
        "ANOMALY ANALYTICS"
    )

    st.header(
        "Anomaly Detection"
    )

    st.caption(
        "Isolation Forest screening of unusual demand "
        "patterns across the 24-hour forecast horizon."
    )

    anomaly_count = int(
        (
            forecast_df[
                "anomaly_prediction"
            ] == -1
        ).sum()
    )

    normal_count = (
        len(forecast_df)
        - anomaly_count
    )

    col1, col2 = st.columns(2)

    col1.metric(
        "Normal Forecast Hours",
        normal_count
    )

    col2.metric(
        "Flagged Anomalies",
        anomaly_count
    )

    anomaly_display = forecast_df.copy()

    anomaly_display["Time"] = (
        pd.to_datetime(
            anomaly_display[
                "timestamp"
            ]
        )
        .dt.strftime(
            "%Y-%m-%d %H:%M"
        )
    )

    anomaly_display[
        "Predicted Demand (kW)"
    ] = (
        anomaly_display[
            "predicted_demand_kW"
        ].round(3)
    )

    anomaly_display[
        "Anomaly Status"
    ] = (
        anomaly_display[
            "anomaly_status"
        ]
    )

    anomaly_display[
        "Peak Risk"
    ] = (
        anomaly_display[
            "peak_risk"
        ]
    )

    st.dataframe(
        anomaly_display[
            [
                "Time",
                "Predicted Demand (kW)",
                "Anomaly Status",
                "Peak Risk"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Isolation Forest is an unsupervised model. "
        "Flags indicate unusual feature patterns rather "
        "than validated anomaly labels."
    )


with tab4:
    st.caption(
        "MODEL EVALUATION"
    )

    st.header(
        "Regression Model Benchmark"
    )

    st.caption(
        "Performance comparison across candidate regression "
        "models using the chronological held-out test period."
    )

    benchmark_df = pd.DataFrame(
        benchmark_data
    )

    st.dataframe(
        benchmark_df,
        use_container_width=True,
        hide_index=True
    )

    xgb_rows = benchmark_df[
        benchmark_df[
            "Model Architecture"
        ]
        .str.contains(
            "XGBoost",
            case=False,
            na=False
        )
    ]

    if not xgb_rows.empty:
        xgb_result = (
            xgb_rows.iloc[0]
        )

        st.subheader(
            "Selected Forecasting Model"
        )

        col1, col2, col3, col4 = (
            st.columns(4)
        )

        col1.metric(
            "RMSE",
            f"{float(xgb_result['RMSE (kW)']):.4f} kW"
        )

        col2.metric(
            "MAE",
            f"{float(xgb_result['MAE (kW)']):.4f} kW"
        )

        col3.metric(
            "R²",
            f"{float(xgb_result['R2 Score']):.4f}"
        )

        col4.metric(
            "MAPE",
            f"{float(xgb_result['MAPE (%)']):.2f}%"
        )

    st.subheader(
        "System Architecture"
    )

    architecture_col1, architecture_col2 = (
        st.columns(2)
    )

    with architecture_col1:
        st.markdown(
            """
**Demand Forecasting**

XGBoost Regressor

**Peak-Risk Classification**

XGBoost Classifier

**Anomaly Detection**

Isolation Forest
"""
        )

    with architecture_col2:
        st.markdown(
            """
**Forecast Horizon**

24 Hours

**Forecast Strategy**

Recursive Multi-Step Forecasting

**Feature Set**

15 Engineered Time-Series Features
"""
        )

    st.warning(
        "MAPE should be interpreted cautiously because "
        "percentage error can become large when actual "
        "electricity demand is relatively small."
    )


with tab5:
    st.caption(
        "MODEL MONITORING"
    )

    st.header(
        "Prediction History"
    )

    st.caption(
        "Stored forecasting records generated through "
        "the dashboard."
    )

    stats = get_database_stats()

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Saved Predictions",
        stats[
            "total_predictions"
        ]
    )

    col2.metric(
        "Saved Peak Warnings",
        stats[
            "total_peak_predictions"
        ]
    )

    col3.metric(
        "Saved Anomaly Flags",
        stats[
            "total_anomalies"
        ]
    )

    history = get_prediction_history(
        limit=100
    )

    history_columns = [
        "ID",
        "Logged At",
        "Forecast Time",
        "Predicted Demand (kW)",
        "Lower Bound (kW)",
        "Upper Bound (kW)",
        "Peak Probability",
        "Peak Prediction",
        "Peak Risk",
        "Anomaly Prediction",
        "Anomaly Status",
        "Actual Demand (kW)",
        "Prediction Error (kW)"
    ]

    if history:
        history_df = pd.DataFrame(
            history,
            columns=history_columns
        )

        history_df[
            "Peak Probability (%)"
        ] = (
            history_df[
                "Peak Probability"
            ] * 100
        ).round(2)

        history_df = (
            history_df.drop(
                columns=[
                    "Peak Probability"
                ]
            )
        )

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

    else:
        st.info(
            "No prediction records have been saved yet. "
            "Save a forecast to begin building prediction history."
        )