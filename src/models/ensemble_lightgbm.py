
"""
LightGBMModel implementation for personality prediction project.
Supports hyperparameter tuning via RandomizedSearchCV.
"""

from src.models import BaseModel
from sklearn.model_selection import RandomizedSearchCV
from lightgbm import LGBMClassifier

class LightGBMModel(BaseModel):
    """
    LightGBM classifier model with optional hyperparameter tuning.

    Args:
        param_grid (dict, optional): Grid of hyperparameters for RandomizedSearchCV.
    """
    def __init__(self, param_grid=None):
        """
        Initialize the LightGBMModel.

        Args:
            param_grid (dict, optional): Grid of hyperparameters for tuning.
        """
        super().__init__(name="LightGBM")
        self.param_grid = param_grid

    def build_model(self):
        """
        Build the LightGBM model. If param_grid is provided, use RandomizedSearchCV for hyperparameter tuning.
        """
        base_model = LGBMClassifier(random_state=self.random_state)
        # Hyperparameter tuning
        if self.param_grid:
            self.model = RandomizedSearchCV(
                estimator=base_model,
                param_distributions=self.param_grid,
                n_iter=20,
                scoring='f1_weighted',
                cv=self.cv,
                n_jobs=-1,
                random_state=self.random_state
            )
        else:
            self.model = base_model