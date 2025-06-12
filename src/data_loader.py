import os
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
pd.options.mode.chained_assignment = None

os.chdir("..")

class DataLoader:

    def __init__(self, raw_path = 'data/raw/personality_dataset.csv', processed_path="data/processed/data_clean.pkl", missing_strategy="fill", random_state=42):
        self.raw_path = raw_path
        self.processed_path = processed_path
        self.scaler = StandardScaler()
        self.missing_strategy = missing_strategy
        self.random_state = random_state
        
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
        return

    def _prepare_and_save(self):
        self.data_raw = self._load_raw()
        self.X, self.y = self._preprocess()
        self._save_processed(self.X, self.y)

        return self.X, self.y

    def _preprocess(self):
        df = self.data_raw.copy()

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

        # Feature encoding
        df['Stage_fear'] = df['Stage_fear'].map({'Yes':1,'No':0})
        df['Drained_after_socializing'] = df['Drained_after_socializing'].map({'Yes':1,'No':0})
        df['Personality'] = df['Personality'].map({'Introvert':1,'Extrovert':0})

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
            extrovert_mean = df[df.Personality=='Extrovert'][col].mean()
            df.loc[df.Personality == 'Extrovert', col] = df.loc[df.Personality == 'Extrovert', col].fillna(extrovert_mean)
            introvert_mean = df[df.Personality=='Introvert'][col].mean()
            df.loc[df.Personality == 'Introvert', col] = df.loc[df.Personality == 'Introvert', col].fillna(introvert_mean)
        
        for col in cat_cols:
            extrovert_mode = df[df.Personality=='Extrovert'][col].mode()[0]
            df.loc[df.Personality == 'Extrovert', col] = df.loc[df.Personality == 'Extrovert', col].fillna(extrovert_mode)
            introvert_mode = df[df.Personality=='Introvert'][col].mode()[0]
            df.loc[df.Personality == 'Introvert', col] = df.loc[df.Personality == 'Introvert', col].fillna(introvert_mode)

        return df

    def _drop_missing(self, df):
        df = df.dropna()
        return df
    
    def get_data_train_test(self, scaled=True, test_size=0.2):
        
        if scaled:
            X = pd.DataFrame(self.scaler.fit_transform(self.X), columns=self.X.columns)
        else :
            X = self.X
        
        X_train, X_test, y_train, y_test = train_test_split(X, self.y, test_size=test_size, random_state=self.random_state)
        return X_train, X_test, y_train, y_test
    
    def get_data_imputed(self):
        return self.data_imputed
