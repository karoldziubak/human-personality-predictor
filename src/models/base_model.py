from abc import ABC, abstractmethod
from sklearn.metrics import accuracy_score

class BaseModel(ABC):
    def __init__(self, name):
        self.name = name
        self.model = None

    @abstractmethod
    def build_model(self):
        pass

    def train(self, X_train, y_train):
        self.build_model()
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        return {"accuracy": acc}
