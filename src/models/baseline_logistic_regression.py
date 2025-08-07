
"""
LogisticRegressionModel implementation for personality prediction project.
Supports hyperparameter tuning via RandomizedSearchCV.
"""

from src.models import BaseModel
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression

class LogisticRegressionModel(BaseModel):
    """
    Logistic Regression classifier model with optional hyperparameter tuning.

    Args:
        param_grid (dict, optional): Grid of hyperparameters for RandomizedSearchCV.
    """
    def __init__(self, param_grid=None):
        """
        Initialize the LogisticRegressionModel.

        Args:
            param_grid (dict, optional): Grid of hyperparameters for tuning.
        """
        super().__init__(name="Logistic Regression")
        self.param_grid = param_grid

    def build_model(self):
        """
        Build the Logistic Regression model. If param_grid is provided, use RandomizedSearchCV for hyperparameter tuning.
        """
        self.model = LogisticRegression(random_state=self.random_state)
        # Hyperparameter tuning
        if self.param_grid:
            self.model = RandomizedSearchCV(
                estimator=self.model,
                param_distributions=self.param_grid,
                n_iter=self.n_iter,
                scoring='f1_weighted',
                cv=self.cv,
                n_jobs=-1,
                random_state=self.random_state
            )
        else:
            self.model = self.model