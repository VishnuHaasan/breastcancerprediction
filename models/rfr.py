from sklearn.ensemble import RandomForestClassifier
from .vit import ViTModel
from .cnn import CNNModel
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from metrics.metrics import PerformanceEvaluation

X_COLUMNS = ['cnn_output', 'vit_class_0', 'vit_class_1']

def create_dataset_rfr():

    cnn_ = CNNModel.load("/home/local/ZOHOCORP/vishnu-pt5599/Desktop/BreastCancerPrediction/models/cnn_model.h5")

    vit_ = ViTModel.load("/home/local/ZOHOCORP/vishnu-pt5599/Desktop/BreastCancerPrediction/models/vit_model.h5")

    path = "/home/local/ZOHOCORP/vishnu-pt5599/Desktop/BreastCancerPrediction/balanced_data"

    cnn_predictions = cnn_.predict(path)

    vit_predictions, true_values = vit_.predict(path)

    cnns_ = cnn_predictions.reshape(-1)

    vits_class_0 = vit_predictions[:, 0]

    vits_class_1 = vit_predictions[:, 1]

    df = pd.DataFrame()

    df['cnn_output'] = cnns_
    df['vit_class_0'] = vits_class_0
    df['vit_class_1'] = vits_class_1
    df['target'] = true_values

    X = df[X_COLUMNS]
    y = df['target']

    return X, y

class RfClassifier:

    def __init__(self) -> None:
        
        self.classifier = RandomForestClassifier()

    def fit(self, save=True):

        X, y = create_dataset_rfr()

        X_train, X_test, y_train, y_test = train_test_split(X, y)

        self.classifier.fit(X_train, y_train)

        predictions = self.classifier.predict(X_test)

        print(PerformanceEvaluation.evaluate(y_pred=predictions, y_true=y_test))

        if save:

            with open('/home/local/ZOHOCORP/vishnu-pt5599/Desktop/BreastCancerPrediction/models/rfr_model.h5', 'wb') as f:
                pickle.dump(self.classifier, f)

        self.fitted_ = True

        return self

    def predict(self, test_path):

        print(self.__dict__)

        if not hasattr(self, 'fitted_'):

            raise ValueError("This instance is not fitted yet, try calling the fit function")

        cnn_ = CNNModel.load("/home/local/ZOHOCORP/vishnu-pt5599/Desktop/BreastCancerPrediction/models/cnn_model.h5")

        vit_ = ViTModel.load("/home/local/ZOHOCORP/vishnu-pt5599/Desktop/BreastCancerPrediction/models/vit_model.h5")

        cnn_prediction = cnn_.predict(test_path)

        vit_prediction, _ = vit_.predict(test_path)

        cnn_prediction = cnn_prediction.reshape(-1)

        vit_class_0 = vit_prediction[:, 0]

        vit_class_1 = vit_prediction[:, 1]

        df = pd.DataFrame()
        
        df['cnn_output'] = cnn_prediction
        df['vit_class_0'] = vit_class_0
        df['vit_class_1'] = vit_class_1

        X = df[X_COLUMNS]

        predictions = self.classifier.predict(X)

        print(predictions)

        return predictions

    @staticmethod
    def load(path):

        with open(path, 'rb') as f:
            model = pickle.load(f)

        classif = RfClassifier()
        classif.classifier = model
        classif.fitted_ = True

        return classif