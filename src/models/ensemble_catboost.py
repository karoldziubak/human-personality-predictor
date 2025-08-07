
"""
CatBoostModel implementation for personality prediction project.
Supports hyperparameter tuning via RandomizedSearchCV.
"""

from src.models import BaseModel
from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostClassifier

class CatBoostModel(BaseModel):
    """
    CatBoost classifier model with optional hyperparameter tuning.

    Args:
        param_grid (dict, optional): Grid of hyperparameters for RandomizedSearchCV.
    """
    def __init__(self, param_grid=None):
        """
        Initialize the CatBoostModel.

        Args:
            param_grid (dict, optional): Grid of hyperparameters for tuning.
        """
        super().__init__(name="CatBoost")
        self.param_grid = param_grid

    def build_model(self):
        """
        Build the CatBoost model. If param_grid is provided, use RandomizedSearchCV for hyperparameter tuning.
        """
        base_model = CatBoostClassifier(random_state=self.random_state)
        # Hyperparameter tuning
        if self.param_grid:
            self.model = RandomizedSearchCV(
                estimator=base_model,
                param_distributions=self.param_grid,
                n_iter=self.n_iter,
                scoring='f1_weighted',
                cv=self.cv,
                n_jobs=-1,
                random_state=self.random_state
            )
        else:
            self.model = base_model