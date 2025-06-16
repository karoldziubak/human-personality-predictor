import os
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
pd.options.mode.chained_assignment = None

os.chdir("..")

class DataLoader:

    def __init__(self, raw_path = 'data/raw/personality_dataset.csv', missing_strategy="fill", random_state=42):
        self.raw_path = raw_path
        self.scaler = StandardScaler()
        self.missing_strategy = missing_strategy
        self.random_state = random_state
        
        self.data_raw = self._load_raw()
        self.X, self.y = self._preprocess()

    def _load_raw(self):
        return pd.read_csv(self.raw_path)

    def _preprocess(self):
        df = self.data_raw.copy()

        df['Stage_fear'] = df['Stage_fear'].map({'Yes':1,'No':0})
        df['Drained_after_socializing'] = df['Drained_after_socializing'].map({'Yes':1,'No':0})
        df['Personality'] = df['Personality'].map({'Introvert':1,'Extrovert':0})

        # Drop duplicates
        df = df.drop_duplicates()

        # Missing values
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
        # Separate columns
        cols = df.columns
        num_cols = [x for x in df.columns if df[x].dtypes != 'O']
        cat_cols = [y for y in cols if y not in num_cols]

        # Fill missing values, numeric with mean, categorical with mode
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
        df = df.dropna()
        return df
    
    def get_data_train_test(self, scaled=True, test_size=0.2):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=self.random_state)
        
        if scaled:
            X_train = pd.DataFrame(self.scaler.fit_transform(X_train), columns=X_train.columns)
            X_test = pd.DataFrame(self.scaler.transform(X_test), columns=X_test.columns)

        return X_train, X_test, y_train, y_test
    
    def get_data_imputed(self):
        return self.data_imputed
