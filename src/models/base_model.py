
"""
BaseModel abstract class for personality prediction models.
Defines the interface and common evaluation/visualization methods for all models.
"""

from abc import ABC, abstractmethod
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class BaseModel(ABC):
    """
    Abstract base class for all personality prediction models.
    Provides interface for model building, training, prediction, evaluation, and visualization.

    Attributes:
        name (str): Name of the model.
        model: The underlying ML model instance.
        random_state (int): Random seed for reproducibility.
        cv (int): Number of cross-validation folds.
        n_iter (int): Number of iterations for hyperparameter search.
    """
    def __init__(self, name):
        """
        Initialize the BaseModel.

        Args:
            name (str): Name of the model.
        """
        self.name = name
        self.model = None
        self.random_state = 42
        self.cv = 5
        self.n_iter = 40

    @abstractmethod
    def build_model(self):
        """
        Build the underlying ML model. Must be implemented by subclasses.
        """
        pass

    def train(self, X_train, y_train):
        """
        Build and fit the model to the training data.

        Args:
            X_train (pd.DataFrame or np.ndarray): Training features.
            y_train (pd.Series or np.ndarray): Training labels.
        """
        self.build_model()
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        """
        Predict labels and probabilities for the test data.

        Args:
            X_test (pd.DataFrame or np.ndarray): Test features.
        Returns:
            tuple: (y_pred, y_proba) - predicted labels and probabilities for class 1.
        """
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]
        return y_pred, y_proba

    def evaluate(self, X_test, y_test, verbose=False):
        """
        Evaluate the model on test data and optionally plot results.

        Args:
            X_test (pd.DataFrame or np.ndarray): Test features.
            y_test (pd.Series or np.ndarray): Test labels.
            verbose (bool): If True, plot evaluation metrics.
        Returns:
            dict: Dictionary with accuracy, AUC, and average precision.
        """
        y_pred, y_proba = self.predict(X_test)
        self.acc = accuracy_score(y_test, y_pred)
        self.auc = roc_auc_score(y_test, y_proba)
        self.ap = average_precision_score(y_test, y_proba)

        if verbose:
            self.plot_results(y_test, y_pred, y_proba)
        return {"accuracy": self.acc, "auc": self.auc, "ap": self.ap}
    
    def plot_results(self, y_test, y_pred, y_proba):
        """
        Plot confusion matrix, ROC curve, and precision-recall curve for model evaluation.

        Args:
            y_test (array-like): True labels.
            y_pred (array-like): Predicted labels.
            y_proba (array-like): Predicted probabilities for class 1.
        """
        plt.figure(figsize=(18,5))

        # Confusion matrix
        plt.subplot(1,3,1)
        y_test_mapped = np.array(list(map(lambda x: 'Extrovert' if x == 0 else 'Introvert', y_test)))
        y_pred_mapped = np.array(list(map(lambda x: 'Extrovert' if x == 0 else 'Introvert', y_pred)))
        print(classification_report(y_test_mapped, y_pred_mapped))
        sns.heatmap(confusion_matrix(y_test_mapped, y_pred_mapped, labels=['Extrovert', 'Introvert']), annot=True, fmt="d", cmap="Blues", xticklabels=['Extrovert', 'Introvert'], yticklabels=['Extrovert', 'Introvert'])
        plt.title(f'Confusion Matrix for {self.name} model')
        plt.xlabel('Predicted')
        plt.ylabel('True')

        # ROC curve
        plt.subplot(1,3,2)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {self.auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve for {self.name} model')
        plt.legend(loc="lower right")

        # Precision-recall curve
        plt.subplot(1,3,3)
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        plt.plot(recall, precision, label=f'PR Curve (AP = {self.ap:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve for {self.name} model')
        plt.legend(loc="lower left")

        plt.show()