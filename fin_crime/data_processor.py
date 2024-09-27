from pathlib import Path
import io
import os
import requests
import zipfile

import geopandas as gpd
import pandas as pd


class DataProcessor:

    date_cols = [
        "trans_date_trans_time",
        "dob"
    ]

    us_coords_url = 'https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_state_500k.zip'

    def __init__(self,
                 output_dir: str,
                 cache_dir: str = "cache"):
        """Initializes the class."""
        self.output_dir = output_dir
        self.cache_dir = cache_dir

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)

        self._download_us_states()

    def _download_us_states(self):
        """Downloads coordinates of US states and saves them to cache"""
        response = requests.get(self.us_coords_url)

        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            z.extractall(os.path.join(self.cache_dir, 'states_shp'))

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

    def process(self, df: pd.DataFrame):
        """Process a DataFrame of raw data"""
        # Load US geography
        df_usa = gpd.read_file(os.path.join(self.cache_dir, 'states_shp/cb_2018_us_state_500k.shp'))

        # Hour of the transaction
        if "trans_date_trans_time" in df.columns:
            df["trans_hour"] = df["trans_date_trans_time"].dt.hour

        # Age of the client at transaction
        if all(True for x in ["trans_date_trans_time", "dob"] if x in df.columns):
            df["age_at_transaction"] = (df["trans_date_trans_time"] - df["dob"]) / pd.Timedelta("365d")

        # State of cardholder
        if all(True for x in ["long", "lat"] if x in df.columns):
            dfg_card = gpd.GeoDataFrame(
                df[['trans_num', 'long', 'lat']],
                geometry=gpd.points_from_xy(df['long'], df['lat']),
                crs="EPSG:4269"
            )
            df["card_state"] = gpd.sjoin(dfg_card, df_usa[['STUSPS', 'geometry']], how='left')["STUSPS"]
            df["card_state"] = df["card_state"].fillna("NaN")

        # State of merchant
        if all(True for x in ["merch_long", "merch_lat"] if x in df.columns):
            dfg_merch = gpd.GeoDataFrame(
                df[['trans_num', 'merch_long', 'merch_lat']],
                geometry=gpd.points_from_xy(df['merch_long'], df['merch_lat']),
                crs="EPSG:4269"
            )
            df["merchant_state"] = gpd.sjoin(dfg_merch, df_usa[['STUSPS', 'geometry']], how='left')["STUSPS"]
            df["merchant_state"] = df["merchant_state"].fillna("NaN")

        return df
