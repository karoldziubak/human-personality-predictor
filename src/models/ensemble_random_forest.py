
"""
RandomForestModel implementation for personality prediction project.
Supports hyperparameter tuning via RandomizedSearchCV.
"""

from src.models import BaseModel
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

class RandomForestModel(BaseModel):
    """
    Random Forest classifier model with optional hyperparameter tuning.

    Args:
        param_grid (dict, optional): Grid of hyperparameters for RandomizedSearchCV.
    """
    def __init__(self, param_grid=None):
        """
        Initialize the RandomForestModel.

        Args:
            param_grid (dict, optional): Grid of hyperparameters for tuning.
        """
        super().__init__(name="Random Forest")
        self.param_grid = param_grid

    def build_model(self):
        """
        Build the Random Forest model. If param_grid is provided, use RandomizedSearchCV for hyperparameter tuning.
        """
        base_model = RandomForestClassifier(random_state=self.random_state)
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