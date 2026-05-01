import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

import warnings
warnings.filterwarnings('ignore')

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras import backend as K
from keras.constraints import Constraint
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping


class OrthogonalWeightConstraint(Constraint):
    def __init__(self, encoding_dim, weightage=1.0, axis=0):
        self.encoding_dim = encoding_dim
        self.weightage = weightage
        self.axis = axis

    def __call__(self, w):
        weights = K.transpose(w) if self.axis == 1 else w
        if self.encoding_dim > 1:
            diff = K.dot(K.transpose(weights), weights) - K.eye(self.encoding_dim)
            return self.weightage * K.sqrt(K.sum(K.square(diff)))
        return K.sum(weights ** 2) - 1.0


def build_autoencoder(input_dim, dropout_rate, learning_rate, bottleneck_name):
    net = Sequential()
    net.add(Dropout(dropout_rate, input_shape=(input_dim,)))
    net.add(Dense(300, activation='relu', use_bias=True,
                  kernel_regularizer=OrthogonalWeightConstraint(300, weightage=1., axis=0)))
    net.add(Dense(50, activation='linear', name=bottleneck_name))
    net.add(Dense(300, activation='relu'))
    net.add(Dense(input_dim, activation='sigmoid'))
    net.compile(loss='mean_squared_error', optimizer=Adam(lr=learning_rate))
    return net


def train_and_encode(model, data, args, bottleneck_name, output_path, condition_labels):
    model.summary()

    early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
    history = model.fit(
        data, data,
        batch_size=args.batch_size,
        epochs=args.epochs,
        shuffle=True,
        verbose=1,
        validation_split=args.validation_split,
        callbacks=[early_stop]
    )

    print("\nTraining Accuracy:   ", history.history['loss'][-1])
    print("Validation Accuracy: ", history.history['val_loss'][-1], "\n")

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validate'], loc='upper right')
    plt.show()

    encoder = Model(model.input, model.get_layer(bottleneck_name).output)
    latent = encoder.predict(data)

    result = pd.DataFrame(latent, index=data.index)
    result['condition'] = condition_labels.values
    result.to_csv(output_path, sep=',')


def load_domain(filepath):
    df = pd.read_csv(filepath, sep=',', index_col=0)
    condition = df.iloc[:, -1]
    features = df.iloc[:, :-1].astype(float)
    return features, condition


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file1',      type=str,   required=True)
    parser.add_argument('--input_file2',      type=str,   required=True)
    parser.add_argument('--output_file1',     type=str,   required=True)
    parser.add_argument('--output_file2',     type=str,   required=True)
    parser.add_argument('--dropout_rate',     type=float, required=True,  default=0.2)
    parser.add_argument('--learning_rate',    type=float, required=True,  default=0.0001)
    parser.add_argument('--batch_size',       type=int,   required=True,  default=16)
    parser.add_argument('--epochs',           type=int,   required=True,  default=200)
    parser.add_argument('--validation_split', type=float, required=False, default=0.2)
    return parser.parse_args()


def main():
    args = parse_args()

    X, X_cond = load_domain(args.input_file1)
    model_A = build_autoencoder(X.shape[1], args.dropout_rate, args.learning_rate, bottleneck_name='bottleneck1')
    train_and_encode(model_A, X, args, 'bottleneck1', args.output_file1, X_cond)

    Y, Y_cond = load_domain(args.input_file2)
    model_B = build_autoencoder(Y.shape[1], args.dropout_rate, args.learning_rate, bottleneck_name='bottleneck2')
    train_and_encode(model_B, Y, args, 'bottleneck2', args.output_file2, Y_cond)


if __name__ == '__main__':
    main()