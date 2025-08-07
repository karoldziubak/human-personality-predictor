
"""
Module for loading and preprocessing personality data.
Contains the DataLoader class for loading, cleaning, and splitting the data.
"""

import os
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
pd.options.mode.chained_assignment = None

os.chdir("..")

class DataLoader:
    """
    Class for loading and preprocessing personality data.

    Args:
        raw_path (str): Path to the raw CSV data file.
        missing_strategy (str): Strategy for handling missing values ('fill' or 'drop').
        random_state (int): Random seed.
    """


    def __init__(self, raw_path = 'data/raw/personality_dataset.csv', missing_strategy="fill", random_state=42):
        """
        Initializes the DataLoader, loads the data, and performs initial preprocessing.

        Args:
            raw_path (str): Path to the raw CSV data file.
            missing_strategy (str): Strategy for handling missing values ('fill' or 'drop').
            random_state (int): Random seed.
        """
        self.raw_path = raw_path
        self.scaler = StandardScaler()
        self.missing_strategy = missing_strategy
        self.random_state = random_state
        self.data_raw = self._load_raw()
        self.X, self.y = self._preprocess()


    def _load_raw(self):
        """
        Loads data from the CSV file.

        Returns:
            pd.DataFrame: Raw data.
        """
        return pd.read_csv(self.raw_path)


    def _preprocess(self):
        """
        Processes data: value mapping, duplicate removal, missing value handling.

        Returns:
            tuple: (X, y) - features and labels.
        """
        df = self.data_raw.copy()
        df['Stage_fear'] = df['Stage_fear'].map({'Yes':1,'No':0})
        df['Drained_after_socializing'] = df['Drained_after_socializing'].map({'Yes':1,'No':0})
        df['Personality'] = df['Personality'].map({'Introvert':1,'Extrovert':0})
        df = df.drop_duplicates()
        preprocessing_dict = {
            "fill": self._fill_missing,
            "drop": self._drop_missing
        }
        if self.missing_strategy in preprocessing_dict:
            df = preprocessing_dict[self.missing_strategy](df)
        else:
            raise ValueError(f"Unknown missing strategy: {self.missing_strategy}. Must be one of {list(preprocessing_dict.keys())}")
        self.data_imputed = df.copy()
        y = df["Personality"]
        X = df.drop(columns=["Personality"])
        return X, y


    def _fill_missing(self, df):
        """
        Fills missing values: numeric columns with class mean, categorical with class mode.

        Args:
            df (pd.DataFrame): Input data.
        Returns:
            pd.DataFrame: Data with missing values filled.
        """
        cols = df.columns
        num_cols = [x for x in df.columns if df[x].dtypes != 'O']
        cat_cols = [y for y in cols if y not in num_cols]
        for col in num_cols:
            extrovert_mean = df[df.Personality==0][col].mean()
            df.loc[df.Personality == 0, col] = df.loc[df.Personality == 0, col].fillna(extrovert_mean)
            introvert_mean = df[df.Personality==1][col].mean()
            df.loc[df.Personality == 1, col] = df.loc[df.Personality == 1, col].fillna(introvert_mean)
        for col in cat_cols:
            extrovert_mode = df[df.Personality==0][col].mode()[0]
            df.loc[df.Personality == 0, col] = df.loc[df.Personality == 0, col].fillna(extrovert_mode)
            introvert_mode = df[df.Personality==1][col].mode()[0]
            df.loc[df.Personality == 1, col] = df.loc[df.Personality == 1, col].fillna(introvert_mode)
        return df


    def _drop_missing(self, df):
        """
        Removes rows with missing values.

        Args:
            df (pd.DataFrame): Input data.
        Returns:
            pd.DataFrame: Data without missing values.
        """
        df = df.dropna()
        return df
    

    def get_data_train_test(self, scaled=True, test_size=0.2):
        """
        Returns train/test split and optionally scales features.

        Args:
            scaled (bool): Whether to scale features.
            test_size (float): Proportion of the test set.
        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=self.random_state)
        if scaled:
            X_train = pd.DataFrame(self.scaler.fit_transform(X_train), columns=X_train.columns)
            X_test = pd.DataFrame(self.scaler.transform(X_test), columns=X_test.columns)
        return X_train, X_test, y_train, y_test

    def get_data_imputed(self):
        """
        Returns data after imputation (after preprocessing).

        Returns:
            pd.DataFrame: Data after imputation.
        """
        return self.data_imputed
