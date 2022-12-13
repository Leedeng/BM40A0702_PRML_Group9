from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization
from tensorflow.keras.layers import Add, AveragePooling2D, Flatten, Dense, MaxPooling2D, ZeroPadding2D,Dropout
from tensorflow.keras.models import Model
import tensorflow as tf
import tensorflow_addons as tfa
'''
This is a simple implementation of ResNet(https://arxiv.org/abs/1512.03385)
'''

def conv2d_batchnorm(x,num_filters,k_size,strides=(1,1),padding='same'):
    '''
    Conv2D + BatchNormalization layer
    '''
    x = Conv2D(num_filters,k_size,strides=strides,activation='relu',padding=padding)(x)
    x = BatchNormalization(axis=3)(x)
    return x

def identity_block(inputs,num_filters,k_size,strides=(1,1),with_shortcut=False):
    x = conv2d_batchnorm(inputs,num_filters=num_filters,k_size=k_size,strides=strides,padding='same')
    if with_shortcut:
        shortcut = conv2d_batchnorm(inputs,num_filters=num_filters,k_size=k_size,strides=strides)
        x = Add()([x,shortcut])
        return x
    else:
        x = Add()([x,inputs])
        return x

def resnet_18(input_shape,learning_rate,dropout_rate,num_class,num_hidden):
    inputs = Input(shape=input_shape)
    x = ZeroPadding2D((3,3))(inputs)
    
    #conv1
    x = conv2d_batchnorm(x,num_filters=32,k_size=(7,7),strides=(2,2),padding='valid')
    x = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')(x)
    
    #conv2_x
    x = identity_block(x,num_filters=32,k_size=(3,3))
    x = identity_block(x,num_filters=32,k_size=(3,3))
    
    #conv3_x4
    x = identity_block(x,num_filters=64,k_size=(3,3),strides=(2,2),with_shortcut=True)
    x = identity_block(x,num_filters=64,k_size=(3,3))
    
    #conv4_x
    x = identity_block(x,num_filters=128,k_size=(3,3),strides=(2,2),with_shortcut=True)
    x = identity_block(x,num_filters=128,k_size=(3,3))
    4
    #conv5_x
    x = identity_block(x,num_filters=256,k_size=(3,3),strides=(2,2),with_shortcut=True)
    x = identity_block(x,num_filters=256,k_size=(3,3))
    
    x = AveragePooling2D(pool_size=(7,7))(x)
    x = Flatten()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(num_hidden, activation="relu")(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(num_class, activation="softmax")(x)

    model = Model(inputs=inputs,outputs=outputs)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )
    return model

class SupervisedContrastiveLoss(tf.keras.losses.Loss):
    def __init__(self, temperature=1, name=None):
        super(SupervisedContrastiveLoss, self).__init__(name=name)
        self.temperature = temperature

    def __call__(self, labels, feature_vectors, sample_weight=None):
        # Normalize feature vectors
        feature_vectors_normalized = tf.math.l2_normalize(feature_vectors, axis=1)
        # Compute logits
        logits = tf.divide(
            tf.matmul(
                feature_vectors_normalized, tf.transpose(feature_vectors_normalized)
            ),
            self.temperature,
        )
        return tfa.losses.npairs_loss(tf.squeeze(labels,axis=1), logits)

def resnet_18_encoder(input_shape,learning_rate,dropout_rate,num_class,num_hidden,temperature,projection_units):
    inputs = Input(shape=input_shape)
    x = ZeroPadding2D((3,3))(inputs)
    
    #conv1
    x = conv2d_batchnorm(x,num_filters=32,k_size=(7,7),strides=(2,2),padding='valid')
    x = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')(x)
    
    #conv2_x
    x = identity_block(x,num_filters=32,k_size=(3,3))
    x = identity_block(x,num_filters=32,k_size=(3,3))
    
    #conv3_x4
    x = identity_block(x,num_filters=64,k_size=(3,3),strides=(2,2),with_shortcut=True)
    x = identity_block(x,num_filters=64,k_size=(3,3))
    
    #conv4_x
    x = identity_block(x,num_filters=128,k_size=(3,3),strides=(2,2),with_shortcut=True)
    x = identity_block(x,num_filters=128,k_size=(3,3))
    4
    #conv5_x
    x = identity_block(x,num_filters=256,k_size=(3,3),strides=(2,2),with_shortcut=True)
    x = identity_block(x,num_filters=256,k_size=(3,3))
    
    x = AveragePooling2D(pool_size=(7,7))(x)
    outputs = Flatten()(x)

    model = Model(inputs=inputs,outputs=outputs)

    return model