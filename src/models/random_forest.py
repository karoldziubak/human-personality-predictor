from src.models import BaseModel
from sklearn.ensemble import RandomForestClassifier

class RandomForestModel(BaseModel):
    def __init__(self):
        super().__init__(name="Random Forest")

    def build_model(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
