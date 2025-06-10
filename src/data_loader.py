import os
import pickle
from sklearn.preprocessing import StandardScaler
import pandas as pd
pd.options.mode.chained_assignment = None

os.chdir("..")

class DataLoader:
    def __init__(self, raw_path = 'data/raw/personality_dataset.csv', processed_path="data/processed/data_clean.pkl", scale=True):
        self.raw_path = raw_path
        self.processed_path = processed_path
        self.scale = scale
        self.scaler = StandardScaler()
        
        if os.path.exists(self.processed_path) and os.path.exists(self.raw_path):
            self.X, self.y = self._load_processed()
            self.data_raw = self._load_raw()
        else:
            self.X, self.y = self._prepare_and_save()

    def _load_raw(self):
        return pd.read_csv(self.raw_path)
    
    def _load_processed(self):
        with open(self.processed_path, "rb") as f:
            return pickle.load(f)

    def _save_processed(self, X, y):
        os.makedirs(os.path.dirname(self.processed_path), exist_ok=True)
        with open(self.processed_path, "wb") as f:
            pickle.dump((X, y), f)

    def _save_raw(self, df):
        os.makedirs(os.path.dirname(self.raw_path), exist_ok=True)
        df.to_csv(self.raw_path, index=False)

    def _prepare_and_save(self):
        df = self._load_raw()
        self.data_raw = df
        self._save_raw(df)

        X, y = self._preprocess(df)

        if self.scale:
            X = pd.DataFrame(self.scaler.fit_transform(X), columns=X.columns)

        self._save_processed(X, y)

        return X, y

    def _preprocess(self, df):
        df = df.dropna()

        df["Stage_fear"] = df["Stage_fear"].map({"Yes": 1, "No": 0})
        df["Drained_after_socializing"] = df["Drained_after_socializing"].map({"Yes": 1, "No": 0})
        df["Personality"] = df["Personality"].map({"Introvert": 0, "Extrovert": 1})

        y = df["Personality"]
        X = df.drop(columns=["Personality"])

        return X, y

    def get_data(self):
        return self.X, self.y
    
