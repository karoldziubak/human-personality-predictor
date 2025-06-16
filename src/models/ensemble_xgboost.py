from src.models import BaseModel
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier

class XGBoostModel(BaseModel):
    def __init__(self, param_grid=None):
        super().__init__(name="XGBoost")
        self.param_grid = param_grid
    
    def build_model(self):
        self.model = XGBClassifier(random_state=self.random_state)

        # Hyperparameter tuning
        if self.param_grid:
            self.model = RandomizedSearchCV(estimator=self.model, param_distributions=self.param_grid, n_iter=20, scoring='f1_weighted', cv=3, n_jobs=-1, random_state=self.random_state)
        else:
            self.model = self.model