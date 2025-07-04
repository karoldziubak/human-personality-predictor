from src.models.ensemble_random_forest import RandomForestModel
from src.models.baseline_logistic_regression import LogisticRegressionModel
from src.models.ensemble_xgboost import XGBoostModel
from src.models.ensemble_lightgbm import LightGBMModel
from src.models.ensemble_catboost import CatBoostModel
from src.models.baseline_decision_tree import DecisionTreeModel
import numpy as np

model_parameters = {
    'random_forest': {
        'model': RandomForestModel(param_grid={'n_estimators': [100, 150, 200, 250, 300], 'max_depth': [None, 2, 4, 6, 8, 10], 'min_samples_split': [3, 5, 7, 9, 11]}),
        'scaled': False,
        },
    'xgboost': {
        'model': XGBoostModel(param_grid={'n_estimators': [100, 150, 200, 250, 300], 'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3], 'max_depth': [None, 2, 4, 6, 8, 10], 'subsample': [0.5, 0.8, 1.0]}),
        'scaled': False,
        },
    'light_gbm': {
        'model': LightGBMModel(param_grid={'n_estimators': [100, 150, 200, 250, 300], 'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3], 'max_depth': [None, 2, 4, 6, 8, 10], 'subsample': [0.5, 0.8, 1.0], 'colsample_bytree': [0.5, 0.8, 1.0], 'num_leaves': np.linspace(15, 100, 5).astype(int).tolist(), 'verbose': [-1]}),
        'scaled': False,
        },
    'catboost':{
        'model': CatBoostModel(param_grid={'n_estimators': [100, 150, 200, 250, 300], 'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3], 'depth': [None, 2, 4, 6, 8, 10], 'l2_leaf_reg': [1, 3, 5, 7, 10], 'verbose': [0]}),
        'scaled': False
    },
    'logistic_regression': {
        'model': LogisticRegressionModel(param_grid={'C': np.logspace(-4, 4, 20), 'solver': ['lbfgs','liblinear']}),
        'scaled': True,
        },
    'decision_tree': {
        'model': DecisionTreeModel(param_grid={'criterion': ['gini', 'entropy', 'log_loss'], 'max_depth': [None, 2, 4, 6, 8, 10], 'min_samples_split': [3, 5, 7, 9, 11], 'min_samples_leaf': [1, 3, 5, 7] ,'max_features': [None, 'sqrt', 'log2'], 'ccp_alpha': [0.0, 0.01, 0.02, 0.05, 0.1]}),
        'scaled': False,
        },
}