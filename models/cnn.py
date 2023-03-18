from keras.models import Sequential, load_model
from keras.layers import (
    Convolution2D,
    MaxPooling2D,
    Flatten,
    Dense
)
from keras.preprocessing.image import ImageDataGenerator

class CNNModel:

    def __init__(self) -> None:
        classifier = Sequential()

        classifier.add(Convolution2D(32, 3, 3, input_shape = (50, 50, 3), activation='relu'))
        classifier.add(MaxPooling2D(pool_size= (2, 2)))

        classifier.add(Convolution2D(32, 3, 3, activation='relu'))
        classifier.add(MaxPooling2D(pool_size= (2, 2)))

        classifier.add(Flatten())

        classifier.add(Dense(128, activation='relu'))
        classifier.add(Dense(1, activation='sigmoid'))

        classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        self.classifier = classifier

    def fit(self,
        train_path, 
        test_path, 
        save=True
    ):

        train_datagen = ImageDataGenerator(rescale = 1./255,
                                    shear_range = 0.2,
                                    zoom_range = 0.2,
                                    horizontal_flip = True)

        test_datagen = ImageDataGenerator(rescale = 1./255)

        training_set = train_datagen.flow_from_directory(train_path,
                                                        target_size = (50, 50),
                                                        batch_size = 32,
                                                        class_mode = 'binary')

        test_set = test_datagen.flow_from_directory(test_path,
                                                    target_size = (50, 50),
                                                    batch_size = 32,
                                                    class_mode = 'binary')

        self.classifier.fit_generator(training_set,
                                steps_per_epoch = 5030//32,
                                epochs = 25,
                                validation_data = test_set,
                                validation_steps = 1759,
                                verbose=1)

        if save:

            self.classifier.save('/home/local/ZOHOCORP/vishnu-pt5599/Desktop/BreastCancerPrediction/models/cnn_model.h5')

        self.fitted_ = True

        return self

    @staticmethod
    def load(path):

        m = load_model(path)

        classif = CNNModel()

        classif.classifier = m

        classif.fitted_ = True

        return classif

    def predict(
        self, 
        path='/home/local/ZOHOCORP/vishnu-pt5599/Desktop/BreastCancerPrediction/balanced_data'
    ):

        if not hasattr(self, 'fitted_'):

            raise ValueError("This instance is not fitted yet, try calling the fit function")

        datagen = ImageDataGenerator(rescale= 1./255)

        data = datagen.flow_from_directory(path, target_size=(50, 50), class_mode='binary')

        predictions = self.classifier.predict(data)

        return predictions
