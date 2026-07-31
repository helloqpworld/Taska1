import os
import sys

import pandas as pd

from features import (calculate_distances, calculate_geometry,
                      generate_advanced_features)


def pipeline_eda_preparation(df: pd.DataFrame) -> pd.DataFrame:
    # EDA
    if "id" in df.columns:
        del df["id"]
    df["vendor_id"] = df["vendor_id"].astype("uint8")
    if "dropoff_datetime" in df.columns:
        del df["dropoff_datetime"]
    df["passenger_count"] = df["passenger_count"].astype("uint8")
    df["store_and_fwd_flag"] = (
        df["store_and_fwd_flag"].map({"N": 0, "Y": 1}).astype("uint8")
    )

    # Preparation
    df = calculate_distances(df)
    df = calculate_geometry(df)

    if "pickup_latitude" in df.columns:
        for s in [
            "pickup_latitude",
            "pickup_longitude",
            "dropoff_latitude",
            "dropoff_longitude",
        ]:
            if s in df.columns:
                del df[s]

    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"]).astype(
        "datetime64[s]"
    )
    df = generate_advanced_features(df)

    float_cols = [
        "distance_haversine_km",
        "distance_manhattan_km",
        "distance_euclidean_deg",
        "centroid_latitude",
        "centroid_longitude",
        "delta_latitude",
        "delta_longitude",
        "direction_bearing",
        "estimated_duration_by_speed",
        "estimated_duration_by_speed_log",
    ]
    df[float_cols] = df[float_cols].astype("float32")

    return df


if __name__ == "__main__":
    if len(sys.argv) != 3:
        script_name = os.path.basename(sys.argv[0])
        print(f"Ошибка! Использование: uv run {script_name} <input_path> <output_path>")
        sys.exit(1)

    input_df = pd.read_csv(sys.argv[1])
    processed_df = pipeline_eda_preparation(input_df)
    # processed_df.to_csv(sys.argv[2], index=False, compression='gzip')
    # processed_df.to_csv(sys.argv[2], index=False)
    processed_df.to_parquet(sys.argv[2], index=False, compression="zstd")
