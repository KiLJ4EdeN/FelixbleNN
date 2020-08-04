import tensorflow as tf
import operator
from tensorflow.keras.constraints import max_norm

class FlexibleNN:
  def __init__(self, input_shape=(64, 64, 1), width=3, conv_blocks=2,
               dense_blocks=1, classes=1,
               filter_start=16, filter_size=(2, 2), activation='relu',
               dense_units=512, use_bn=True,
               use_dropout=True, dropout_rate=0.5,
               use_constraint=True, constraint_rate=1,
               use_pool=True, pool_size=(2, 2),
              ):

    # base parameters.
    self.input = tf.keras.layers.Input(shape=input_shape)
    self.width = width
    self.conv_blocks = conv_blocks
    self.dense_blocks = dense_blocks
    self.classes = classes
    self.model = None

    # layer paramters.
    self.filter_start = filter_start
    self.filter_size = filter_size
    self.activation = activation
    self.dense_units = dense_units
    self.use_bn = use_bn

    # regularization.
    self.use_dropout = use_dropout
    self.dropout_rate = dropout_rate   
    self.use_constraint = use_constraint
    self.constraint_rate = constraint_rate
    self.use_pool = use_pool
    self.pool_size = pool_size
    
    # non user related.
    self.outputs = None

  def add_tuples(self, a, b):
    res = tuple(map(operator.add, a, b))
    return res

  def build_model(self):
    # do a combination of blocks here.
    x = self.__parallel_block(self.input)
    x = self.__dense_block(x)
    x = self.__classification_block(x)
    self.output = x
    self.model = tf.keras.models.Model(inputs=self.input,
                                  outputs=self.output)
    return self.model

  def __parallel_block(self, x):
    branches = []
    for i in range(self.width):
      f = self.__conv_block(x, i)
      f = tf.keras.layers.Flatten()(f)
      branches.append(f)
    x = tf.keras.layers.concatenate(branches)
    return x

  def __residual_block(self, x):
    # residual block.
    x_shortcut = x
    # Path 1
    x = tf.keras.layers.Conv2D(self.filter_start*1,
                                       (1, 1), strides = (2, 2))(X)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(self.activation)(x)
    # Path 2
    x = tf.keras.layers.Conv2D(filters = self.filter_start*1,
                                       kernel_size = self.filter_size, 
                                       strides = (1,1), padding = 'same')(X)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(self.activation)(X)
    # Path 3
    x = tf.keras.layers.Conv2D(filters = self.filter_start*2,
                                       kernel_size = (1, 1),
                                       strides = (1,1), padding = 'valid')(X)
    x = tf.keras.layers.BatchNormalization()(x)
    # Shortcut
    x_shortcut = tf.keras.layers.Conv2D(filters = self.filter_start*2,
                                                kernel_size = (1, 1),
                                                strides = (2, 2),
                                                padding = 'valid')(x_shortcut)
    x_shortcut = tf.keras.layers.BatchNormalization()(x_shortcut)
    # Final Path.
    x = tf.keras.layers.Add()([x, x_shortcut])
    x = tf.keras.layers.Activation(self.activation)(x)
    return x

  def __identity_block(self, x):
    # resnet identity block.
    x_shortcut = x
    # Path 1
    x = tf.keras.layers.Conv2D(filters = self.filter_start*1, kernel_size = (1, 1),
                               strides = (1, 1), padding = 'valid')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(self.activation)(x)
    # Path 2
    x = tf.keras.layers.Conv2D(filters = self.filter_start*2, kernel_size = self.filter_size,
                               strides = (1, 1), padding = 'same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(self.activation)(x)
    # Path 3
    x = tf.keras.layers.Conv2D(filters = x.shape[2], kernel_size = (1, 1),
                               strides = (1, 1), padding = 'valid')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    # Final Path.
    x = tf.keras.layers.Add()([x, x_shortcut])
    x = tf.keras.layers.Activation(self.activation)(x)
    return x

  def __conv_block(self, x, r):

    if self.use_constraint:
      constraint = max_norm(self.constraint_rate)
    else:
      constraint = None
    for i in range(self.conv_blocks):
      x = tf.keras.layers.Conv2D(filters=self.filter_start*(r+1),
                                          kernel_size=self.add_tuples(self.filter_size, (r+1, r+1)), 
                                          kernel_constraint=constraint)(x)
      if self.use_bn:
        x = tf.keras.layers.BatchNormalization()(x)
      x = tf.keras.layers.Activation(self.activation)(x)
      x = tf.keras.layers.Conv2D(filters=self.filter_start*(r+1),
                                          kernel_size=self.add_tuples(self.filter_size, (r+1, r+1)), 
                                          kernel_constraint=constraint)(x)
      if self.use_bn:
        x = tf.keras.layers.BatchNormalization()(x)
      x = tf.keras.layers.Activation(self.activation)(x)
      if self.use_dropout:
        x = tf.keras.layers.Dropout(self.dropout_rate)(x)
      x = tf.keras.layers.MaxPooling2D(pool_size=self.pool_size)(x)
    return x

  def __dense_block(self, x):

    if self.use_constraint:
      constraint = max_norm(self.constraint_rate)
    else:
      constraint = None
    # add some dense layers.
    for i in range(self.dense_blocks):
      x = tf.keras.layers.Dense(self.dense_units,
                                activation=self.activation,
                                kernel_constraint=constraint)(x)
      if self.use_dropout:
        x = tf.keras.layers.Dropout(self.dropout_rate)(x)
    return x

  def __classification_block(self, x):
    # classification layer.
    if self.classes > 1:
      x = tf.keras.layers.Dense(self.classes, activation='softmax')(x)
    else:
      x = tf.keras.layers.Dense(self.classes, activation='sigmoid')(x)
    return x
