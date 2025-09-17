import tensorflow as tf
from keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
import pickle

class Autoencoder:
    def __init__(self, input_neurons=33, input_size=33, embedding_size=200,
                 optimizer='adam', metrics=['CosineSimilarity'], loss=tf.keras.losses.mean_squared_logarithmic_error):
        #hyperparameters
        self.optimizer = optimizer
        self.metrics = metrics
        self.loss = loss
        self.initializer = tf.keras.initializers.GlorotUniform(seed=0)
        #create the nn
        self.autoencoder = tf.keras.models.Sequential()
        self.autoencoder.add(tf.keras.layers.Dense(input_neurons, input_shape=(input_size,)))
        self.autoencoder.add(tf.keras.layers.Dropout(0.3, seed=0))
        #encoder
        encoder = tf.keras.layers.Dense(embedding_size, activation=tf.nn.sigmoid, kernel_initializer=self.initializer)
        self.autoencoder.add(encoder)
        #decoder
        decoder = tf.keras.layers.Dense(input_size, activation=tf.nn.sigmoid, kernel_initializer=self.initializer)
        self.autoencoder.add(decoder)
        #save the layers indexes to encode and decode later
        self.index_last_encoder_layer = self.autoencoder.layers.index(encoder)
        self.index_decoder_layer = self.autoencoder.layers.index(decoder)
        self.autoencoder.summary()

    def save_initialize_weights(self, initialize_weights_file='assets'):
        weights = self.autoencoder.get_weights()
        # Guardar en archivo
        with open(initialize_weights_file, "wb") as f:
            pickle.dump(weights, f)


    def fit(self, X_train, y_train, X_val, y_val, epochs=200, batch_size=256, shuffle=False):
        self.autoencoder.compile(optimizer=self.optimizer, metrics=self.metrics, loss=self.loss)
        # Train the nn
        history = self.autoencoder.fit(X_train, y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        validation_data=(X_val, y_val))
        return history
    
    def save(self, name='model.h5'):
        self.autoencoder.save(name)

    def load_model(self, name='model.h5'):
        self.autoencoder = load_model(name)
        return self.autoencoder
   
    def predict(self, X):
        y_pred = self.autoencoder.predict(X)
        return y_pred

    def encode(self):
        self.encoder = Model(inputs=self.autoencoder.input, outputs=self.autoencoder.get_layer(index=self.index_last_encoder_layer).output)
        return self.encoder

    def decode(self):
        self.decoder = Model(inputs=self.autoencoder.get_layer(index=self.index_last_encoder_layer).output, outputs=self.autoencoder.get_layer(index=self.index_decoder_layer).output)
        return self.decoder