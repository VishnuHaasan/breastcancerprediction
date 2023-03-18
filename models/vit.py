import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras import layers

num_classes=2
input_shape=(50, 50, 3)

path = "/home/local/ZOHOCORP/vishnu-pt5599/Desktop/BreastCancerPrediction/balanced_data"

datagen = ImageDataGenerator()

train_generator = datagen.flow_from_directory(
        path,
        batch_size=5030,
        target_size=(50, 50),
        class_mode='binary')

x, y = train_generator.next()

learning_rate = 0.001
weight_decay = 0.0001
batch_size = 256
num_epochs = 100
image_size = 72
patch_size = 6
num_patches = (image_size // patch_size) ** 2
projection_dim = 64
num_heads = 4
transformer_units = [
    projection_dim * 2,
    projection_dim
]
transformer_layers = 8
mlp_head_units = [2048, 1024]

data_augmentation = keras.Sequential(
    [
        layers.Normalization(),
        layers.Resizing(image_size, image_size),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(factor=0.02),
        layers.RandomZoom(
            height_factor=0.2, width_factor=0.2
        ),
    ],
    name="data_augmentation",
)
data_augmentation.layers[0].adapt(x)

def MLP(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

class Patches(layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        
    def get_config(self):
        config = super().get_config()
        config['patch_size'] = self.patch_size
        return config

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )
        
    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) - self.position_embedding(positions)
        return encoded

def create_vit_classifier():
    inputs = layers.Input(shape=input_shape)
    augmented = data_augmentation(inputs)
    patches = Patches(patch_size)(augmented)
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)
    
    for _ in range(transformer_layers):
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        
        x2 = layers.Add()([attention_output, encoded_patches])
        
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        
        x3 = MLP(x3, hidden_units=transformer_units, dropout_rate=0.1)
        
        encoded_patches = layers.Add()([x3, x2])
        
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    
    features = MLP(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    
    logits = layers.Dense(num_classes, activation=tf.keras.activations.softmax)(features)
    
    model = keras.Model(inputs=inputs, outputs=logits)
    
    return model

def run_experiment(model):
    optimizer = tfa.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
        ],
    )

    checkpoint_filepath = "/tmp/checkpoint"
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
    )

    history = model.fit(
        x=x,
        y=y,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_split=0.1,
        callbacks=[checkpoint_callback],
    )

    model.save('/home/local/ZOHOCORP/vishnu-pt5599/Desktop/BreastCancerPrediction/models/vit_model.h5')

    return history

def load_model_vit():

    m = load_model('/home/local/ZOHOCORP/vishnu-pt5599/Desktop/BreastCancerPrediction/models/vit_model.h5', custom_objects={'Patches': Patches, 'PatchEncoder': PatchEncoder})

    return m

def predict_vit(path, model):

    datagen = ImageDataGenerator()

    data = datagen.flow_from_directory(
        path,
        batch_size=5030,
        target_size=(50, 50),
        class_mode='binary')

    x, y = data.next()

    predictions = model.predict(x)

    return predictions, y

class ViTModel:

    def __init__(self) -> None:
        
        self.classifier = create_vit_classifier()

    def fit(self, path):
    
        optimizer = tfa.optimizers.AdamW(
            learning_rate=learning_rate, weight_decay=weight_decay
        )

        self.classifier.compile(
            optimizer=optimizer,
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[
                keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
                keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
            ],
        )

        checkpoint_filepath = "/tmp/checkpoint"
        checkpoint_callback = keras.callbacks.ModelCheckpoint(
            checkpoint_filepath,
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=True,
        )

        history = self.classifier.fit(
            x=x,
            y=y,
            batch_size=batch_size,
            epochs=num_epochs,
            validation_split=0.1,
            callbacks=[checkpoint_callback],
        )

        print(history)

        self.classifier.save('/home/local/ZOHOCORP/vishnu-pt5599/Desktop/BreastCancerPrediction/models/vit_model.h5')

        self.fitted_ = True

        return self

    def predict(self, path):

        if not hasattr(self, 'fitted_'):

            raise ValueError("This instance is not fitted yet, try calling the fit function")            

        datagen = ImageDataGenerator()

        data = datagen.flow_from_directory(
            path,
            batch_size=5030,
            target_size=(50, 50),
            class_mode='binary')

        x, y = data.next()

        predictions = self.classifier.predict(x)

        return predictions, y

    @staticmethod
    def load(path):

        m = load_model(path, custom_objects={'Patches': Patches, 'PatchEncoder': PatchEncoder})

        classif = ViTModel()

        classif.classifier = m

        classif.fitted_ = True
        
        return classif

