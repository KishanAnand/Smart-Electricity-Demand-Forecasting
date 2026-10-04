import sqlite3
from datetime import datetime

DB_FILE = "energy_analytics.db"



def get_connection():
    return sqlite3.connect(DB_FILE)



def init_db():
    """
    Creates the prediction_logs table if it does not exist.
    """

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS prediction_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,

            timestamp TEXT NOT NULL,

            forecast_timestamp TEXT,

            predicted_demand REAL NOT NULL,

            lower_bound REAL,

            upper_bound REAL,

            peak_probability REAL,

            peak_prediction INTEGER,

            peak_risk TEXT,

            anomaly_prediction INTEGER,

            anomaly_status TEXT,

            actual_demand REAL,

            prediction_error REAL
        )
    """)

    conn.commit()
    conn.close()



def ensure_columns():
    """
    Adds missing columns to an older prediction_logs table.

    This lets you continue using your existing
    energy_analytics.db instead of deleting it.
    """

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "PRAGMA table_info(prediction_logs)"
    )

    existing_columns = {
        row[1]
        for row in cursor.fetchall()
    }

    required_columns = {
        "forecast_timestamp": "TEXT",
        "predicted_demand": "REAL",
        "lower_bound": "REAL",
        "upper_bound": "REAL",
        "peak_probability": "REAL",
        "peak_prediction": "INTEGER",
        "peak_risk": "TEXT",
        "anomaly_prediction": "INTEGER",
        "anomaly_status": "TEXT",
        "actual_demand": "REAL",
        "prediction_error": "REAL"
    }

    for column_name, column_type in required_columns.items():

        if column_name not in existing_columns:

            cursor.execute(
                f"""
                ALTER TABLE prediction_logs
                ADD COLUMN {column_name} {column_type}
                """
            )

    conn.commit()
    conn.close()



def log_prediction(
    predicted_demand,
    forecast_timestamp=None,
    lower_bound=None,
    upper_bound=None,
    peak_probability=None,
    peak_prediction=None,
    peak_risk=None,
    anomaly_prediction=None,
    anomaly_status=None,
    actual_demand=None
):
    """
    Stores one forecast record.

    prediction_error is only calculated when actual_demand
    is available.
    """

    conn = get_connection()
    cursor = conn.cursor()

    logged_at = datetime.now().isoformat(
        timespec="seconds"
    )

    prediction_error = None

    if actual_demand is not None:

        prediction_error = (
            float(actual_demand)
            - float(predicted_demand)
        )

    cursor.execute("""
        INSERT INTO prediction_logs (
            timestamp,
            forecast_timestamp,
            predicted_demand,
            lower_bound,
            upper_bound,
            peak_probability,
            peak_prediction,
            peak_risk,
            anomaly_prediction,
            anomaly_status,
            actual_demand,
            prediction_error
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        logged_at,

        (
            str(forecast_timestamp)
            if forecast_timestamp is not None
            else None
        ),

        float(predicted_demand),

        (
            float(lower_bound)
            if lower_bound is not None
            else None
        ),

        (
            float(upper_bound)
            if upper_bound is not None
            else None
        ),

        (
            float(peak_probability)
            if peak_probability is not None
            else None
        ),

        (
            int(peak_prediction)
            if peak_prediction is not None
            else None
        ),

        peak_risk,

        (
            int(anomaly_prediction)
            if anomaly_prediction is not None
            else None
        ),

        anomaly_status,

        (
            float(actual_demand)
            if actual_demand is not None
            else None
        ),

        (
            float(prediction_error)
            if prediction_error is not None
            else None
        )
    ))

    conn.commit()
    conn.close()



def get_prediction_history(limit=100):
    """
    Returns latest prediction records.
    """

    limit = max(
        1,
        int(limit)
    )

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            id,
            timestamp,
            forecast_timestamp,
            predicted_demand,
            lower_bound,
            upper_bound,
            peak_probability,
            peak_prediction,
            peak_risk,
            anomaly_prediction,
            anomaly_status,
            actual_demand,
            prediction_error

        FROM prediction_logs

        ORDER BY id DESC

        LIMIT ?
    """, (limit,))

    rows = cursor.fetchall()

    conn.close()

    return rows


def get_database_stats():

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT COUNT(*) FROM prediction_logs"
    )

    total_predictions = cursor.fetchone()[0]

    cursor.execute("""
        SELECT COUNT(*)
        FROM prediction_logs
        WHERE peak_prediction = 1
    """)

    total_peak_predictions = cursor.fetchone()[0]

    cursor.execute("""
        SELECT COUNT(*)
        FROM prediction_logs
        WHERE anomaly_prediction = -1
    """)

    total_anomalies = cursor.fetchone()[0]

    cursor.execute("""
        SELECT AVG(ABS(prediction_error))
        FROM prediction_logs
        WHERE prediction_error IS NOT NULL
    """)

    result = cursor.fetchone()

    average_absolute_error = (
        result[0]
        if result and result[0] is not None
        else None
    )

    conn.close()

    return {
        "total_predictions":
            total_predictions,

        "total_peak_predictions":
            total_peak_predictions,

        "total_anomalies":
            total_anomalies,

        "average_absolute_error":
            average_absolute_error
    }



def clear_prediction_history():

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "DELETE FROM prediction_logs"
    )

    conn.commit()
    conn.close()



init_db()
ensure_columns()


if __name__ == "__main__":

    print(
        "Database initialized successfully:"
    )

    print(DB_FILE)

    print(
        get_database_stats()
    )