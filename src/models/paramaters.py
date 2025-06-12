from src.models.random_forest import RandomForestModel
from src.models.logistic_regression import LogisticRegressionModel
import numpy as np

model_parameters = {
    'random_forest': {
        'model': RandomForestModel(param_grid={'n_estimators': [100, 150, 200], 'max_depth': [None, 2, 4, 6, 8, 10, 12], 'min_samples_split': [5, 7, 9, 11]}),
        'scaled': False,
        },
    'logistic_regression': {
        'model': LogisticRegressionModel(param_grid={'C': np.logspace(-4, 4, 20), 'solver': ['lbfgs','liblinear']}),
        'scaled': True,
        },
}