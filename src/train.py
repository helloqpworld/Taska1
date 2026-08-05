import os
import sys

import dotenv

dotenv.load_dotenv()

import lightgbm as lgb
import matplotlib.pyplot as plt
import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split

mlflow.set_tracking_uri("http://localhost:5050")
mlflow.set_experiment("taxi_duration_lightgbm")
mlflow.lightgbm.autolog()


def train(df: pd.DataFrame) -> pd.DataFrame:
    features_to_drop = [
        "pickup_datetime",
        "vendor_passenger_interaction",
        "trip_duration",
    ]

    X = df.drop(columns=[col for col in features_to_drop if col in df.columns])
    y = df["trip_duration"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    limit = y_train.quantile(0.99)
    mask = y_train <= limit
    X_train_clean = X_train[mask]
    y_train_clean = y_train[mask]

    y_train_log = np.log1p(y_train_clean)

    lgb_model = lgb.LGBMRegressor(
        n_estimators=15, learning_rate=0.1, random_state=42, n_jobs=-1, verbose=-1
    )

    with mlflow.start_run(run_name="lgb_with_minio_storage") as run:
        print(f"Эксперимент начат! Run ID: {run.info.run_id}")

        lgb_model.fit(X_train_clean, y_train_log)

        y_pred_log = lgb_model.predict(X_test)

        y_test_original = y_test.values
        y_pred_original = np.expm1(y_pred_log)

        mae = mean_absolute_error(y_test_original, y_pred_original)
        r2 = r2_score(y_test_original, y_pred_original)
        rmse = root_mean_squared_error(y_test_original, y_pred_original)

        mlflow.log_metric("val_mae", mae)
        mlflow.log_metric("val_r2", r2)
        mlflow.log_metric("val_rmse", rmse)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(y_test_original, y_pred_original, alpha=0.3, color="blue")
        ax.plot(
            [y_test_original.min(), y_test_original.max()],
            [y_test_original.min(), y_test_original.max()],
            "r--",
            lw=2,
        )
        ax.set_xlabel("Actual Trip Duration")
        ax.set_ylabel("Predicted Trip Duration")
        ax.set_title("Predicted vs Actual (Original Scale)")
        plt.tight_layout()

        plot_path = "predicted_vs_actual.png"
        plt.savefig(plot_path)
        plt.close(fig)
        mlflow.log_artifact(plot_path)
        os.remove(plot_path)

        print(f"Эксперимент завершен! Run ID: {run.info.run_id}")

        output_df = pd.DataFrame(
            {
                "actual_trip_duration": y_test_original,
                "predicted_trip_duration": y_pred_original,
            }
        )
        return output_df


if __name__ == "__main__":
    if len(sys.argv) != 3:
        script_name = os.path.basename(sys.argv[0])
        print(
            f"Ошибка! Использование: uv run {script_name} <input_path_data> <output_path_prediction>"
        )
        sys.exit(1)

    input_df = pd.read_parquet(sys.argv[1])
    predicts_df = train(input_df)

    predicts_df.to_parquet(sys.argv[2], index=False, compression="zstd")
    print(f"Результаты успешно сохранены в {sys.argv[2]}")
