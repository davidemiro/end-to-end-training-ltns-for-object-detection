# -*- coding: utf-8 -*-
'''ResNet50 model for Keras.
# Reference:
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
Adapted from code contributed by BigMoyan.
'''

from __future__ import print_function
from __future__ import absolute_import

from keras.layers import Input, Add, Dense, Activation, Flatten, Convolution2D, MaxPooling2D, ZeroPadding2D, \
    AveragePooling2D, TimeDistributed

from keras import backend as K

from keras_frcnn.RoiPoolingConv import RoiPoolingConv
from keras_frcnn.FixedBatchNormalization import FixedBatchNormalization
from keras_frcnn.bb_creation import bb_creation,bb_creation_v1,bb_creation_v2
from keras_frcnn import ltn
from keras.layers.merge import concatenate
from keras_frcnn.Clause import Clause,Literal_Clause
import tensorflow as tf

import keras
from imaplib import Literal

def clause_Layer(input):
    

    
    result = tf.minimum(1.0,tf.reduce_sum(input,1, keep_dims=True))
    h = tf.div(tf.to_float(tf.size(result)),tf.reduce_sum(tf.reciprocal(result),keep_dims=True))
    return h

def get_weight_path():
    if K.image_dim_ordering() == 'th':
        return 'resnet50_weights_th_dim_ordering_th_kernels_notop.h5'
    else:
        return 'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

def get_img_output_length(width, height):
    def get_output_length(input_length):
        # zero_pad
        input_length += 6
        # apply 4 strided convolutions
        filter_sizes = [7, 3, 1, 1]
        stride = 2
        for filter_size in filter_sizes:
            input_length = (input_length - filter_size + stride) // stride
        return input_length

    return get_output_length(width), get_output_length(height) 

def identity_block(input_tensor, kernel_size, filters, stage, block, trainable=True):

    nb_filter1, nb_filter2, nb_filter3 = filters
    
    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Convolution2D(nb_filter1, (1, 1), name=conv_name_base + '2a', trainable=trainable)(input_tensor)
    x = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter2, (kernel_size, kernel_size), padding='same', name=conv_name_base + '2b', trainable=trainable)(x)
    x = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter3, (1, 1), name=conv_name_base + '2c', trainable=trainable)(x)
    x = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = Add()([x, input_tensor])
    x = Activation('relu')(x)
    return x


def identity_block_td(input_tensor, kernel_size, filters, stage, block, trainable=True):

    # identity block time distributed

    nb_filter1, nb_filter2, nb_filter3 = filters
    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = TimeDistributed(Convolution2D(nb_filter1, (1, 1), trainable=trainable, kernel_initializer='normal'), name=conv_name_base + '2a')(input_tensor)
    x = TimeDistributed(FixedBatchNormalization(axis=bn_axis), name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = TimeDistributed(Convolution2D(nb_filter2, (kernel_size, kernel_size), trainable=trainable, kernel_initializer='normal',padding='same'), name=conv_name_base + '2b')(x)
    x = TimeDistributed(FixedBatchNormalization(axis=bn_axis), name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = TimeDistributed(Convolution2D(nb_filter3, (1, 1), trainable=trainable, kernel_initializer='normal'), name=conv_name_base + '2c')(x)
    x = TimeDistributed(FixedBatchNormalization(axis=bn_axis), name=bn_name_base + '2c')(x)

    x = Add()([x, input_tensor])
    x = Activation('relu')(x)

    return x

def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2), trainable=True):

    nb_filter1, nb_filter2, nb_filter3 = filters
    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Convolution2D(nb_filter1, (1, 1), strides=strides, name=conv_name_base + '2a', trainable=trainable)(input_tensor)
    x = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter2, (kernel_size, kernel_size), padding='same', name=conv_name_base + '2b', trainable=trainable)(x)
    x = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter3, (1, 1), name=conv_name_base + '2c', trainable=trainable)(x)
    x = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Convolution2D(nb_filter3, (1, 1), strides=strides, name=conv_name_base + '1', trainable=trainable)(input_tensor)
    shortcut = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x


def conv_block_td(input_tensor, kernel_size, filters, stage, block, input_shape, strides=(2, 2), trainable=True):

    # conv block time distributed

    nb_filter1, nb_filter2, nb_filter3 = filters
    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = TimeDistributed(Convolution2D(nb_filter1, (1, 1), strides=strides, trainable=trainable, kernel_initializer='normal'), input_shape=input_shape, name=conv_name_base + '2a')(input_tensor)
    x = TimeDistributed(FixedBatchNormalization(axis=bn_axis), name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = TimeDistributed(Convolution2D(nb_filter2, (kernel_size, kernel_size), padding='same', trainable=trainable, kernel_initializer='normal'), name=conv_name_base + '2b')(x)
    x = TimeDistributed(FixedBatchNormalization(axis=bn_axis), name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = TimeDistributed(Convolution2D(nb_filter3, (1, 1), kernel_initializer='normal'), name=conv_name_base + '2c', trainable=trainable)(x)
    x = TimeDistributed(FixedBatchNormalization(axis=bn_axis), name=bn_name_base + '2c')(x)

    shortcut = TimeDistributed(Convolution2D(nb_filter3, (1, 1), strides=strides, trainable=trainable, kernel_initializer='normal'), name=conv_name_base + '1')(input_tensor)
    shortcut = TimeDistributed(FixedBatchNormalization(axis=bn_axis), name=bn_name_base + '1')(shortcut)

    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x

def nn_base(input_tensor=None, trainable=False):

    # Determine proper input shape
    if K.image_dim_ordering() == 'th':
        input_shape = (3, None, None)
    else:
        input_shape = (None, None, 3)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1

    x = ZeroPadding2D((3, 3))(img_input)

    x = Convolution2D(64, (7, 7), strides=(2, 2), name='conv1', trainable = trainable)(x)
    x = FixedBatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), trainable = trainable)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b', trainable = trainable)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c', trainable = trainable)

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', trainable = trainable)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b', trainable = trainable)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c', trainable = trainable)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d', trainable = trainable)

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', trainable = trainable)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b', trainable = trainable)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c', trainable = trainable)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d', trainable = trainable)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e', trainable = trainable)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f', trainable = trainable)

    return x


def classifier_layers(x, input_shape, trainable=False):

    # compile times on theano tend to be very high, so we use smaller ROI pooling regions to workaround
    # (hence a smaller stride in the region that follows the ROI pool)
    if K.backend() == 'tensorflow':
        x = conv_block_td(x, 3, [512, 512, 2048], stage=5, block='a', input_shape=input_shape, strides=(2, 2), trainable=trainable)
    elif K.backend() == 'theano':
        x = conv_block_td(x, 3, [512, 512, 2048], stage=5, block='a', input_shape=input_shape, strides=(1, 1), trainable=trainable)

    x = identity_block_td(x, 3, [512, 512, 2048], stage=5, block='b', trainable=trainable)
    x = identity_block_td(x, 3, [512, 512, 2048], stage=5, block='c', trainable=trainable)
    x = TimeDistributed(AveragePooling2D((7, 7)), name='avg_pool')(x)

    return x


def rpn(base_layers,num_anchors):

    x = Convolution2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv1')(base_layers)

    x_class = Convolution2D(num_anchors, (1, 1), activation='sigmoid', kernel_initializer='uniform', name='rpn_out_class')(x)
    x_regr = Convolution2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero', name='rpn_out_regress')(x)

    return [x_class, x_regr, base_layers]




def classifierInputVectorClassNewClause(base_layers, input_rois,num_rois, nb_classes,Y, trainable=False):

    # compile times on theano tend to be very high, so we use smaller ROI pooling regions to workaround
    
    if K.backend() == 'tensorflow':
        pooling_regions = 14
        input_shape = (num_rois,14,14,1024)
    elif K.backend() == 'theano':
        pooling_regions = 7
        input_shape = (num_rois,1024,7,7)

    out_roi_pool = RoiPoolingConv(pooling_regions, num_rois)([base_layers, input_rois])
    out = classifier_layers(out_roi_pool, input_shape=input_shape, trainable=True)

    out = TimeDistributed(Flatten())(out)
    out_class = TimeDistributed(Dense(nb_classes, activation='softmax', kernel_initializer='zero'), name='dense_class_{}'.format(nb_classes))(out)
    # note: no regression target for bg class
    out_regr = TimeDistributed(Dense(4 * (nb_classes-1), activation='linear', kernel_initializer='zero'), name='dense_regress_{}'.format(nb_classes))(out)
    

    output = []
    print(nb_classes)
    for i in range(nb_classes - 1):
        x = ltn.Predicate(num_features=nb_classes,k=6,i=i)(out_class)
        x = Literal_Clause(i)([x,Y[i]])
        x = keras.layers.Lambda(lambda o: tf.Print(o,[o],"Clause {}".format(i)))(x)
        output.append(x)
    out_ltn = keras.layers.Concatenate(axis=1)(output)

    out_ltn = keras.layers.Lambda(lambda x:keras.backend.expand_dims(x,0))(out_ltn)

    
    return [out_regr , out_ltn]

def classifierInputVectorClassNewClauseEvaluate(base_layers, input_rois,num_rois, nb_classes, trainable=False):

    # compile times on theano tend to be very high, so we use smaller ROI pooling regions to workaround
    
    if K.backend() == 'tensorflow':
        pooling_regions = 14
        input_shape = (num_rois,14,14,1024)
    elif K.backend() == 'theano':
        pooling_regions = 7
        input_shape = (num_rois,1024,7,7)

    out_roi_pool = RoiPoolingConv(pooling_regions, num_rois)([base_layers, input_rois])
    out = classifier_layers(out_roi_pool, input_shape=input_shape, trainable=True)

    out = TimeDistributed(Flatten())(out)
    out_class = TimeDistributed(Dense(nb_classes, activation='softmax', kernel_initializer='zero'), name='dense_class_{}'.format(nb_classes))(out)
    # note: no regression target for bg class
    out_regr = TimeDistributed(Dense(4 * (nb_classes-1), activation='linear', kernel_initializer='zero'), name='dense_regress_{}'.format(nb_classes))(out)
    

    output = []
   
    for i in range(nb_classes - 1):
        x = ltn.Predicate(num_features=nb_classes,k=6,i=i)(out_class)
        #x = Literal_Clause(i)([x,Y[i]])
        output.append(x)
    out_ltn = keras.layers.Concatenate(axis=1)(output)

    out_ltn = keras.layers.Lambda(lambda x:keras.backend.expand_dims(x,0))(out_ltn)

    
    return [out_regr ,out_class, out_ltn]
#versione del classificatore che integra anche la parte di regressione nel training
#questo codice identifica l' allenamento che ha ottenuto mAP = 0.31
def classifier(base_layers, input_rois,num_rois, nb_classes,A,B, trainable=False):
    if K.backend() == 'tensorflow':
        pooling_regions = 14
        input_shape = (num_rois,14,14,1024)
    elif K.backend() == 'theano':
        pooling_regions = 7
        input_shape = (num_rois,1024,7,7)
    
    
    #base_layers = tf.Print(base_layers,[base_layers],"BASE_LAYERS_SHAPE")
    
    out_roi_pool = RoiPoolingConv(pooling_regions, num_rois)([base_layers, input_rois])
    out = classifier_layers(out_roi_pool, input_shape=input_shape, trainable=True)

    out = TimeDistributed(Flatten())(out)
    out_class = TimeDistributed(Dense(nb_classes, activation='softmax', kernel_initializer='zero'), name='dense_class_{}'.format(nb_classes))(out)
    # note: no regression target for bg class
    out_regr = TimeDistributed(Dense(4 * (nb_classes-1), activation='linear', kernel_initializer='zero'), name='dense_regress_{}'.format(nb_classes))(out)
    
    
    tensors = bb_creation(nb_classes)([out_regr,out_class])
    

    output = []
    
    #out_class = keras.layers.Lambda(lambda o: tf.Print(o,[o],"out_class"))(out_class)
 
    
    for i in range(nb_classes - 1):
        
        
        
        
        p = ltn.Predicate(num_features=nb_classes+4,k=6,i=i)
        
        x = p(tensors[i])
        
        #x = keras.layers.Lambda(lambda o: tf.Print(o,[o],"predicates {}".format(i)))(x)
        
        
        #questa espressione serve a garantire il ruolo del literal
        import pdb;pdb.set_trace()
        a = keras.layers.Lambda(lambda o: tf.reshape(o,[num_rois,1]))(A[i])
        b = keras.layers.Lambda(lambda o: tf.reshape(o,[num_rois,1]))(B[i])
        
        x = keras.layers.Subtract()([a,x])
        x = keras.layers.Multiply()([x,b])

        x = keras.layers.Lambda(lambda o: clause_Layer(o))(x)
        
        output.append(x)
        
    
    out_ltn = keras.layers.Concatenate(axis=1)(output)
    out_ltn = keras.layers.Lambda(lambda x:keras.backend.expand_dims(x,0))(out_ltn)
    
    
    return [out_regr , out_ltn]
    
def getDims(base_layers):
    if K.image_dim_ordering() == 'th':
        return base_layers.shape[2],base_layers.shape[3]
    else:
        return base_layers.shape[1],base_layers.shape[2]

#Questa versisone del classifier esegue l' allenamento usando come input della LTN il solo out_class

def classifierInputVectorClass(base_layers, input_rois,num_rois, nb_classes,A,B, trainable=False):

    # compile times on theano tend to be very high, so we use smaller ROI pooling regions to workaround
    
    if K.backend() == 'tensorflow':
        pooling_regions = 14
        input_shape = (num_rois,14,14,1024)
    elif K.backend() == 'theano':
        pooling_regions = 7
        input_shape = (num_rois,1024,7,7)

    out_roi_pool = RoiPoolingConv(pooling_regions, num_rois)([base_layers, input_rois])
    out = classifier_layers(out_roi_pool, input_shape=input_shape, trainable=True)

    out = TimeDistributed(Flatten())(out)
    out_class = TimeDistributed(Dense(nb_classes, activation='softmax', kernel_initializer='zero'), name='dense_class_{}'.format(nb_classes))(out)
    # note: no regression target for bg class
    out_regr = TimeDistributed(Dense(4 * (nb_classes-1), activation='linear', kernel_initializer='zero'), name='dense_regress_{}'.format(nb_classes))(out)
    

    output = []

    for i in range(nb_classes - 1):

        
        
        x = ltn.Predicate(num_features=nb_classes,k=6,i=i)(out_class)
        #questa espressione serve a garantire il ruolo del literal
        
        b = keras.layers.Lambda(lambda o: tf.reshape(o,[num_rois,1]))(B[i])
        a = keras.layers.Lambda(lambda o: tf.reshape(o,[num_rois,1]))(A[i])
        
        
        x = keras.layers.Subtract()([a,x])
        x = keras.layers.Multiply()([x,b])
        

        x = Clause(tnorm="luk",aggregator="hmean")(x)
        
        output.append(x)
  
        

    out_ltn = keras.layers.Concatenate(axis=1)(output)
    out_ltn = keras.layers.Lambda(lambda x:keras.backend.expand_dims(x,0))(out_ltn)

    
    return [out_regr , out_ltn]
def classifierInputVectorClass(base_layers, input_rois,num_rois, nb_classes,A,B, trainable=False):

    # compile times on theano tend to be very high, so we use smaller ROI pooling regions to workaround
    
    if K.backend() == 'tensorflow':
        pooling_regions = 14
        input_shape = (num_rois,14,14,1024)
    elif K.backend() == 'theano':
        pooling_regions = 7
        input_shape = (num_rois,1024,7,7)

    out_roi_pool = RoiPoolingConv(pooling_regions, num_rois)([base_layers, input_rois])
    out = classifier_layers(out_roi_pool, input_shape=input_shape, trainable=True)

    out = TimeDistributed(Flatten())(out)
    out_class = TimeDistributed(Dense(nb_classes, activation='softmax', kernel_initializer='zero'), name='dense_class_{}'.format(nb_classes))(out)
    # note: no regression target for bg class
    out_regr = TimeDistributed(Dense(4 * (nb_classes-1), activation='linear', kernel_initializer='zero'), name='dense_regress_{}'.format(nb_classes))(out)
    

    output = []

    for i in range(nb_classes - 1):

        
        
        x = ltn.Predicate(num_features=nb_classes,k=6,i=i)(out_class)
        #questa espressione serve a garantire il ruolo del literal
        
        b = keras.layers.Lambda(lambda o: tf.reshape(o,[num_rois,1]))(B[i])
        a = keras.layers.Lambda(lambda o: tf.reshape(o,[num_rois,1]))(A[i])
        
        
        x = keras.layers.Subtract()([a,x])
        x = keras.layers.Multiply()([x,b])
        

        x = Clause(tnorm="luk",aggregator="hmean")(x)
        
        output.append(x)
  
        

    out_ltn = keras.layers.Concatenate(axis=1)(output)
    out_ltn = keras.layers.Lambda(lambda x:keras.backend.expand_dims(x,0))(out_ltn)

    
    return [out_regr , out_ltn]

#Questa implementazione per generare gli input della LTN fa sempre la "concatenazione" out_class-out_regr in bb_creation ma richiama la bb_creation_v1
#Con questa implementazione ho raggiunto mAP = 0.52  
def classifierNewRegression(base_layers, input_rois,num_rois, nb_classes,A,B, trainable=False):

    # compile times on theano tend to be very high, so we use smaller ROI pooling regions to workaround
    
    if K.backend() == 'tensorflow':
        pooling_regions = 14
        input_shape = (num_rois,14,14,1024)
    elif K.backend() == 'theano':
        pooling_regions = 7
        input_shape = (num_rois,1024,7,7)
    
    out_roi_pool = RoiPoolingConv(pooling_regions, num_rois)([base_layers, input_rois])
    out = classifier_layers(out_roi_pool, input_shape=input_shape, trainable=True)

    out = TimeDistributed(Flatten())(out)
    out_class = TimeDistributed(Dense(nb_classes, activation='softmax', kernel_initializer='zero'), name='dense_class_{}'.format(nb_classes))(out)
    # note: no regression target for bg class
    out_regr = TimeDistributed(Dense(4 * (nb_classes-1), activation='linear', kernel_initializer='zero'), name='dense_regress_{}'.format(nb_classes))(out)
    tensors = bb_creation_v1(nb_classes,num_rois)([out_regr,out_class,input_rois,base_layers])
    output = []
    for i in range(nb_classes - 1):
        
        p = ltn.Predicate(num_features=nb_classes + 4,k=6,i=i)
        
        x = p(tensors[i])
        #questa espressione serve a garantire il ruolo del literal
    
        a = keras.layers.Lambda(lambda o: tf.reshape(o,[num_rois,1]))(A[i])
        b = keras.layers.Lambda(lambda o: tf.reshape(o,[num_rois,1]))(B[i])
        
        x = keras.layers.Subtract()([a,x])
        x = keras.layers.Multiply()([x,b])

        x = keras.layers.Lambda(lambda o: clause_Layer(o))(x)
        
        output.append(x)
        
    
    out_ltn = keras.layers.Concatenate(axis=1)(output)
    out_ltn = keras.layers.Lambda(lambda x:keras.backend.expand_dims(x,0))(out_ltn)
    
    
    return [out_regr , out_ltn]

#Uguale a classifierInputVectorClass ma aggiunge il predicato LTN per la classe background(bg)
def classifierInputVectorClassBgPredicate(base_layers, input_rois,num_rois, nb_classes,A,B, trainable=False):

    # compile times on theano tend to be very high, so we use smaller ROI pooling regions to workaround
    
    if K.backend() == 'tensorflow':
        pooling_regions = 14
        input_shape = (num_rois,14,14,1024)
    elif K.backend() == 'theano':
        pooling_regions = 7
        input_shape = (num_rois,1024,7,7)

    out_roi_pool = RoiPoolingConv(pooling_regions, num_rois)([base_layers, input_rois])
    out = classifier_layers(out_roi_pool, input_shape=input_shape, trainable=True)

    out = TimeDistributed(Flatten())(out)
    out_class = TimeDistributed(Dense(nb_classes, activation='softmax', kernel_initializer='zero'), name='dense_class_{}'.format(nb_classes))(out)
    # note: no regression target for bg class
    out_regr = TimeDistributed(Dense(4 * (nb_classes-1), activation='linear', kernel_initializer='zero'), name='dense_regress_{}'.format(nb_classes))(out)
    

    

    output = []
    
    for i in range(nb_classes):
        
        x = ltn.Predicate(num_features=nb_classes,k=6,i=i)(out_class)
        #questa espressione serve a garantire il ruolo del literal
    
        a = keras.layers.Lambda(lambda o: tf.reshape(o,[num_rois,1]))(A[i])
        b = keras.layers.Lambda(lambda o: tf.reshape(o,[num_rois,1]))(B[i])
        
        x = keras.layers.Subtract()([a,x])
        x = keras.layers.Multiply()([x,b])

        x = keras.layers.Lambda(lambda o: clause_Layer(o))(x)
        
        output.append(x)
        

    out_ltn = keras.layers.Concatenate(axis=1)(output)
    out_ltn = keras.layers.Lambda(lambda x:keras.backend.expand_dims(x,0))(out_ltn)

    
    return [out_regr , out_ltn] 

#uguale a classifierNewRegression ma come prima aggiungo il predicato LTN per la classe bg
def classifierNewRegressionBgPredicate(base_layers, input_rois,num_rois, nb_classes,A,B, trainable=False):

    # compile times on theano tend to be very high, so we use smaller ROI pooling regions to workaround
    
    if K.backend() == 'tensorflow':
        pooling_regions = 14
        input_shape = (num_rois,14,14,1024)
    elif K.backend() == 'theano':
        pooling_regions = 7
        input_shape = (num_rois,1024,7,7)
    
    out_roi_pool = RoiPoolingConv(pooling_regions, num_rois)([base_layers, input_rois])
    out = classifier_layers(out_roi_pool, input_shape=input_shape, trainable=True)

    out = TimeDistributed(Flatten())(out)
    out_class = TimeDistributed(Dense(nb_classes, activation='softmax', kernel_initializer='zero'), name='dense_class_{}'.format(nb_classes))(out)
    # note: no regression target for bg class
    out_regr = TimeDistributed(Dense(4 * (nb_classes-1), activation='linear', kernel_initializer='zero'), name='dense_regress_{}'.format(nb_classes))(out)
    
    #uso la versione v2 di bb_creation che aggiunge alla lista dei tensori quello di input per la classe bg
    tensors = bb_creation_v2(nb_classes,num_rois)([out_regr,out_class,input_rois,base_layers])
    output = []
    for i in range(nb_classes):
        
        p = ltn.Predicate(num_features=nb_classes + 4,k=6,i=i)
        
        x = p(tensors[i])
        #questa espressione serve a garantire il ruolo del literal
    
        a = keras.layers.Lambda(lambda o: tf.reshape(o,[num_rois,1]))(A[i])
        b = keras.layers.Lambda(lambda o: tf.reshape(o,[num_rois,1]))(B[i])
        
        x = keras.layers.Subtract()([a,x])
        x = keras.layers.Multiply()([x,b])

        x = keras.layers.Lambda(lambda o: clause_Layer(o))(x)
        
        output.append(x)
        
    
    out_ltn = keras.layers.Concatenate(axis=1)(output)
    out_ltn = keras.layers.Lambda(lambda x:keras.backend.expand_dims(x,0))(out_ltn)
    
    
    return [out_regr , out_ltn]

#QUI CI SONO TUTTE LE VERSIONI DEL CLASSIFIER DA CHIAMARE IN FASE DI INFERENZA
#DATO CHE IL MODULO LTN CAMBIA IN FASE DI INFERENZA HO DEFINITO QUESTI CLASSIFIER
#IL NOME DI OGNIUNA E' DATO DAL NOME DEL CORRISPETTIVO CLASSIFIER DEL TRAINING+"Evaluate"
#Es classifierInputVectorClass -> classifierInputVectorClassEvaluate
def classifierEvaluate(base_layers, input_rois,num_rois, nb_classes, trainable=False):
    if K.backend() == 'tensorflow':
        pooling_regions = 14
        input_shape = (num_rois,14,14,1024)
    elif K.backend() == 'theano':
        pooling_regions = 7
        input_shape = (num_rois,1024,7,7)
    
    
    #base_layers = tf.Print(base_layers,[base_layers],"BASE_LAYERS_SHAPE")
    
    out_roi_pool = RoiPoolingConv(pooling_regions, num_rois)([base_layers, input_rois])
    out = classifier_layers(out_roi_pool, input_shape=input_shape, trainable=True)

    out = TimeDistributed(Flatten())(out)
    out_class = TimeDistributed(Dense(nb_classes, activation='softmax', kernel_initializer='zero'), name='dense_class_{}'.format(nb_classes))(out)
    # note: no regression target for bg class
    out_regr = TimeDistributed(Dense(4 * (nb_classes-1), activation='linear', kernel_initializer='zero'), name='dense_regress_{}'.format(nb_classes))(out)
    
    
    tensors = bb_creation(nb_classes)([out_regr,out_class])
    output = []
    for i in range(nb_classes - 1):
        p = ltn.Predicate(num_features=nb_classes+4,k=6,i=i)
        x = p(tensors[i])
        #vengono scartati i layer relativi al comportamento del literal e del clause
        output.append(x)
        
    
    out_ltn = keras.layers.Concatenate(axis=1)(output)
    out_ltn = keras.layers.Lambda(lambda x:keras.backend.expand_dims(x,0))(out_ltn)
    
    
    return [out_regr , out_ltn]
def classifierInputVectorClassEvaluate(base_layers, input_rois,num_rois, nb_classes, trainable=False):

    # compile times on theano tend to be very high, so we use smaller ROI pooling regions to workaround
    
    if K.backend() == 'tensorflow':
        pooling_regions = 14
        input_shape = (num_rois,14,14,1024)
    elif K.backend() == 'theano':
        pooling_regions = 7
        input_shape = (num_rois,1024,7,7)

    out_roi_pool = RoiPoolingConv(pooling_regions, num_rois)([base_layers, input_rois])
    out = classifier_layers(out_roi_pool, input_shape=input_shape, trainable=True)

    out = TimeDistributed(Flatten())(out)
    out_class = TimeDistributed(Dense(nb_classes, activation='softmax', kernel_initializer='zero'), name='dense_class_{}'.format(nb_classes))(out)
    # note: no regression target for bg class
    out_regr = TimeDistributed(Dense(4 * (nb_classes-1), activation='linear', kernel_initializer='zero'), name='dense_regress_{}'.format(nb_classes))(out)
    
    

    

    output = []
    
    for i in range(nb_classes - 1):
        
        x = ltn.Predicate(num_features=nb_classes,k=6,i=i)(out_class)
      
      
        output.append(x)
        

    out_ltn = keras.layers.Concatenate(axis=1)(output)
    out_ltn = keras.layers.Lambda(lambda x:keras.backend.expand_dims(x,0))(out_ltn)

    
    return [out_regr , out_ltn]

def classifierNewRegressionEvaluate(base_layers, input_rois,num_rois, nb_classes,trainable=False):

    # compile times on theano tend to be very high, so we use smaller ROI pooling regions to workaround
    
    if K.backend() == 'tensorflow':
        pooling_regions = 14
        input_shape = (num_rois,14,14,1024)
    elif K.backend() == 'theano':
        pooling_regions = 7
        input_shape = (num_rois,1024,7,7)
    
    out_roi_pool = RoiPoolingConv(pooling_regions, num_rois)([base_layers, input_rois])
    out = classifier_layers(out_roi_pool, input_shape=input_shape, trainable=True)

    out = TimeDistributed(Flatten())(out)
    out_class = TimeDistributed(Dense(nb_classes, activation='softmax', kernel_initializer='zero'), name='dense_class_{}'.format(nb_classes))(out)
    # note: no regression target for bg class
    out_regr = TimeDistributed(Dense(4 * (nb_classes-1), activation='linear', kernel_initializer='zero'), name='dense_regress_{}'.format(nb_classes))(out)
    tensors = bb_creation_v1(nb_classes,num_rois)([out_regr,out_class,input_rois,base_layers])
    output = []
    for i in range(nb_classes - 1):
        
        p = ltn.Predicate(num_features=nb_classes + 4,k=6,i=i)
        
        x = p(tensors[i])
        #questa espressione serve a garantire il ruolo del literal
    
        output.append(x)
        
    
    out_ltn = keras.layers.Concatenate(axis=1)(output)
    out_ltn = keras.layers.Lambda(lambda x:keras.backend.expand_dims(x,0))(out_ltn)
    
    
    return [out_regr , out_ltn]

def classifierInputVectorClassBgPredicateEvaluate(base_layers, input_rois,num_rois, nb_classes, trainable=False):

    # compile times on theano tend to be very high, so we use smaller ROI pooling regions to workaround
    
    if K.backend() == 'tensorflow':
        pooling_regions = 14
        input_shape = (num_rois,14,14,1024)
    elif K.backend() == 'theano':
        pooling_regions = 7
        input_shape = (num_rois,1024,7,7)

    out_roi_pool = RoiPoolingConv(pooling_regions, num_rois)([base_layers, input_rois])
    out = classifier_layers(out_roi_pool, input_shape=input_shape, trainable=True)

    out = TimeDistributed(Flatten())(out)
    out_class = TimeDistributed(Dense(nb_classes, activation='softmax', kernel_initializer='zero'), name='dense_class_{}'.format(nb_classes))(out)
    # note: no regression target for bg class
    out_regr = TimeDistributed(Dense(4 * (nb_classes-1), activation='linear', kernel_initializer='zero'), name='dense_regress_{}'.format(nb_classes))(out)
    

    

    output = []
    
    for i in range(nb_classes):
        
        x = ltn.Predicate(num_features=nb_classes,k=6,i=i)(out_class)
        
        output.append(x)
        

    out_ltn = keras.layers.Concatenate(axis=1)(output)
    out_ltn = keras.layers.Lambda(lambda x:keras.backend.expand_dims(x,0))(out_ltn)

    
    return [out_regr , out_ltn]
  
def classifierNewRegressionBgPredicateEvaluate(base_layers, input_rois,num_rois, nb_classes, trainable=False):

    # compile times on theano tend to be very high, so we use smaller ROI pooling regions to workaround
    
    if K.backend() == 'tensorflow':
        pooling_regions = 14
        input_shape = (num_rois,14,14,1024)
    elif K.backend() == 'theano':
        pooling_regions = 7
        input_shape = (num_rois,1024,7,7)
    
    out_roi_pool = RoiPoolingConv(pooling_regions, num_rois)([base_layers, input_rois])
    out = classifier_layers(out_roi_pool, input_shape=input_shape, trainable=True)

    out = TimeDistributed(Flatten())(out)
    out_class = TimeDistributed(Dense(nb_classes, activation='softmax', kernel_initializer='zero'), name='dense_class_{}'.format(nb_classes))(out)
    # note: no regression target for bg class
    out_regr = TimeDistributed(Dense(4 * (nb_classes-1), activation='linear', kernel_initializer='zero'), name='dense_regress_{}'.format(nb_classes))(out)
    tensors = bb_creation_v2(nb_classes,num_rois)([out_regr,out_class,input_rois,base_layers])
    output = []
    for i in range(nb_classes):
        
        p = ltn.Predicate(num_features=nb_classes + 4,k=6,i=i)
        
        x = p(tensors[i])
        
        
        output.append(x)
        
    
    out_ltn = keras.layers.Concatenate(axis=1)(output)
    out_ltn = keras.layers.Lambda(lambda x:keras.backend.expand_dims(x,0))(out_ltn)
    
    
    return [out_regr , out_ltn]

def classifierInputVectorClassNewClauseEvaluate(base_layers, input_rois,num_rois, nb_classes, trainable=False):

    # compile times on theano tend to be very high, so we use smaller ROI pooling regions to workaround
    
    if K.backend() == 'tensorflow':
        pooling_regions = 14
        input_shape = (num_rois,14,14,1024)
    elif K.backend() == 'theano':
        pooling_regions = 7
        input_shape = (num_rois,1024,7,7)

    out_roi_pool = RoiPoolingConv(pooling_regions, num_rois)([base_layers, input_rois])
    out = classifier_layers(out_roi_pool, input_shape=input_shape, trainable=True)

    out = TimeDistributed(Flatten())(out)
    out_class = TimeDistributed(Dense(nb_classes, activation='softmax', kernel_initializer='zero'), name='dense_class_{}'.format(nb_classes))(out)
    # note: no regression target for bg class
    out_regr = TimeDistributed(Dense(4 * (nb_classes-1), activation='linear', kernel_initializer='zero'), name='dense_regress_{}'.format(nb_classes))(out)
    

    output = []
    for i in range(nb_classes - 1):
        x = ltn.Predicate(num_features=nb_classes,k=6,i=i)(out_class)
        output.append(x)
    out_ltn = keras.layers.Concatenate(axis=1)(output)

    out_ltn = keras.layers.Lambda(lambda x:keras.backend.expand_dims(x,0))(out_ltn)

    
    return [out_regr , out_ltn]


