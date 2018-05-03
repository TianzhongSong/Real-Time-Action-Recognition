from keras.layers import Dense, Dropout, Conv2D, Input, MaxPool2D, Flatten, \
   Activation, concatenate
from keras.regularizers import l2
from keras.models import Model


def c2d(input_shape, nb_classes, weight_decay=0.005):
   inputs = Input(input_shape)
   x = Conv2D(16, (3, 3), strides=(1, 1), padding='same',use_bias=True, kernel_regularizer=l2(weight_decay))(inputs)
   x = Activation('relu')(x)

   x = MaxPool2D((2, 2), strides=(2, 2), padding='same')(x)

   x = Conv2D(32, (3, 3), strides=(1, 1), padding='same', use_bias=True, kernel_regularizer=l2(weight_decay))(x)
   x = Activation('relu')(x)
   x = MaxPool2D((2, 2), strides=(2, 2), padding='same')(x)

   x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', use_bias=True,kernel_regularizer=l2(weight_decay))(x)
   x = Activation('relu')(x)
   x = MaxPool2D((2, 2), strides=(2, 2), padding='same')(x)

   x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', use_bias=True, kernel_regularizer=l2(weight_decay))(x)
   x = Activation('relu')(x)
   x = MaxPool2D((2, 2), strides=(2, 2), padding='same')(x)
   x = Dropout(0.25)(x)
   x = Flatten()(x)
   x = Dense(256, activation='relu', kernel_regularizer=l2(weight_decay))(x)
   x = Dropout(0.5)(x)
   x = Dense(nb_classes, kernel_regularizer=l2(weight_decay))(x)
   x = Activation('softmax')(x)

   model = Model(inputs, x)
   return model
