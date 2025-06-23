from abc import ABC, abstractmethod
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class BaseModel(ABC):
    def __init__(self, name):
        self.name = name
        self.model = None
        self.random_state = 42
        self.cv = 5
        self.n_iter = 40

    @abstractmethod
    def build_model(self):
        pass

    def train(self, X_train, y_train):
        self.build_model()
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]
        return y_pred, y_proba

    def evaluate(self, X_test, y_test, verbose=False):
        y_pred, y_proba = self.predict(X_test)
        self.acc = accuracy_score(y_test, y_pred)
        self.auc = roc_auc_score(y_test, y_proba)
        self.ap = average_precision_score(y_test, y_proba)

        if verbose:
            self.plot_results(y_test, y_pred, y_proba)
        return {"accuracy": self.acc, "auc": self.auc, "ap": self.ap}
    
    def plot_results(self, y_test, y_pred, y_proba):
        plt.figure(figsize=(18,5))

        plt.subplot(1,3,1)
        y_test_mapped = np.array(list(map(lambda x: 'Extrovert' if x == 0 else 'Introvert', y_test)))
        y_pred_mapped = np.array(list(map(lambda x: 'Extrovert' if x == 0 else 'Introvert', y_pred)))
        print(classification_report(y_test_mapped, y_pred_mapped))
        sns.heatmap(confusion_matrix(y_test_mapped, y_pred_mapped, labels=['Extrovert', 'Introvert']), annot=True, fmt="d", cmap="Blues", xticklabels=['Extrovert', 'Introvert'], yticklabels=['Extrovert', 'Introvert'])
        plt.title(f'Confusion Matrix for {self.name} model')
        plt.xlabel('Predicted')
        plt.ylabel('True')

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

        #precision-recall curve
        plt.subplot(1,3,3)
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        plt.plot(recall, precision, label=f'PR Curve (AP = {self.ap:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve for {self.name} model')
        plt.legend(loc="lower left")


        plt.show()

