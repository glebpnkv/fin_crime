from pathlib import Path
import os

import pandas as pd


class DataProcessor:

    date_cols = [
        "trans_date_trans_time",
        "dob"
    ]

    def __init__(self,
                 output_dir: str):
        """Initializes the class."""
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def process_raw(self,
                    input_path: str):
        """Processes the raw data and saves the processed data as a parquet file."""
        cur_file_path = Path(input_path)
        cur_file_name = cur_file_path.stem

        # Cannot do anything if the file does not exist
        if not cur_file_path.is_file():
            print(f"File {cur_file_path} does not exist")
            return

        try:
            df_raw = pd.read_csv(cur_file_path, index_col=0)
        except pd.errors.EmptyDataError as e:
            print(f"Unable to read data from file {cur_file_path}: {e}")
            return

        # Formatting date columns
        for col in self.date_cols:
            df_raw[col] = pd.to_datetime(df_raw[col])

        # Writing as parquet
        df_raw.to_parquet(
            path=os.path.join(self.output_dir, f"{cur_file_name}.parquet"),
            index=False
        )

    @staticmethod
    def process(df: pd.DataFrame):
        """Process a DataFrame of raw data"""
        # Hour of the transaction
        if "trans_date_trans_time" in df.columns:
            df["trans_hour"] = df["trans_date_trans_time"].dt.hour

        # Age of the client at transaction
        if all(True for x in ["trans_date_trans_time", "dob"] if x in df.columns):
            df["age_at_transaction"] = (df["trans_date_trans_time"] - df["dob"]) / pd.Timedelta("365d")

        return df
