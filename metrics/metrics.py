from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix

class PerformanceEvaluation:

    @staticmethod
    def evaluate(y_pred, y_true):

        out = dict()

        out['accuracy_score'] = accuracy_score(y_pred=y_pred, y_true=y_true)
        out['f1_score'] = f1_score(y_pred=y_pred, y_true=y_true)
        out['precision_score'] = precision_score(y_pred=y_pred, y_true=y_true)
        out['recall_score'] = recall_score(y_pred=y_pred, y_true=y_true)
        out['roc_auc_score'] = roc_auc_score(y_score=y_pred, y_true=y_true)
        out['confusion_matrix'] = confusion_matrix(y_pred=y_pred, y_true=y_true)

        return out