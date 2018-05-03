"""
define the 3dcnn model, reference: C3D
"""
from keras.layers import Dense, Dropout, Conv3D, Input, MaxPool3D, Flatten, Activation
from keras.regularizers import l2
from keras.models import Model


def c3d_model(input_shape, nb_classes, weight_decay=0.0005):
    inputs = Input(input_shape)
    x = Conv3D(16, (3, 3, 3), strides=(1, 1, 1), padding='same',
               activation='relu', kernel_regularizer=l2(weight_decay))(inputs)
    x = MaxPool3D((2, 2, 1), strides=(2, 2, 1), padding='same')(x)

    x = Conv3D(32, (3, 3, 3), strides=(1, 1, 1), padding='same',
               activation='relu', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPool3D((2, 2, 2), strides=(2, 2, 2), padding='same')(x)

    x = Conv3D(32, (3, 3, 3), strides=(1, 1, 1), padding='same',
               activation='relu', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPool3D((2, 2, 2), strides=(2, 2, 2), padding='same')(x)

    x = Conv3D(64, (3, 3, 3), strides=(1, 1, 1), padding='same',
               activation='relu', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPool3D((2, 2, 2), strides=(2, 2, 2), padding='same')(x)

    x = Conv3D(64, (3, 3, 3), strides=(1, 1, 1), padding='same',
               activation='relu', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPool3D((2, 2, 2), strides=(2, 2, 2), padding='same')(x)

    x = Flatten()(x)
    # x = Dropout(0.25)(x)
    x = Dense(256, activation='relu', kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu', kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(0.5)(x)
    x = Dense(nb_classes, kernel_regularizer=l2(weight_decay))(x)
    x = Activation('softmax')(x)

    model = Model(inputs, x)
    return model
