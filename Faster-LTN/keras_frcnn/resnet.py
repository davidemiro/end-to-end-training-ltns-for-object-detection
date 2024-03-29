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
from keras_frcnn.bb_creation import bb_creation
from keras_frcnn import ltn
from keras.layers.merge import concatenate
from keras_frcnn.Clause import Clause
from keras_frcnn.Clause import Pair
from keras_frcnn.Clause import get_part_whole_ontology
from keras_frcnn.Literal import Literal
from keras_frcnn.Literal import Literal_O
import tensorflow as tf

import keras


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


def get_parts(batch,batch_size):
    output = []
    for p in range(batch_size//2):
        for _ in range(batch_size//2):
            output.append(batch[p,:])
    o = tf.concat(output,axis=0)
    o = tf.expand_dims(o,axis=1)
    return o

def get_wholes(batch,batch_size):
    output = []
    for _ in range(batch_size//2):
        for w in range(batch_size//2):
            output.append(batch[w, :])
    o = tf.concat(output, axis=0)
    o = tf.expand_dims(o, axis=1)
    return o




# Faster-LTN classifier with LTN integration
def classifier(base_layers, input_rois, num_rois, nb_classes ,tnorm , aggregator,activation,gamma,Y,Y_partOf=None,classes=None,std_x=None, std_y=None, std_w=None, std_h=None):

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

    out_class = TimeDistributed(Dense(nb_classes, activation=activation, kernel_initializer='zero'), name='dense_class_{}'.format(nb_classes))(out)
    # note: no regression target for bg class
    out_regr = TimeDistributed(Dense(4 * (nb_classes-1), activation='linear', kernel_initializer='zero'), name='dense_regress_{}'.format(nb_classes))(out)

    output = []
    predictions = {}

    classes = sorted(classes)
    print(nb_classes)

    for i in range(nb_classes):
        p = ltn.Predicate(num_features=nb_classes, k=6, i=i)
        x = p(out_class)
        predictions[classes[i]] = x

        x = Literal(name=str(i),batch_size=num_rois)([x,Y[i]])
        x = Clause(tnorm=tnorm, aggregator=aggregator,gamma=gamma, name = classes[i])(x)
        output.append(x)


    #partOF

    x = bb_creation(nb_classes, num_rois)([out_class, input_rois, base_layers])
    x = Pair(num_rois)(x)
    partOf = ltn.Predicate(num_features=(nb_classes + 5) * 2, k=6, i=nb_classes + 1)
    partOf_prediction = partOf(x)
    x = Literal(name='partOf_literal', batch_size=num_rois//2 * num_rois//2)([partOf_prediction, Y_partOf])
    x = Clause(tnorm=tnorm, aggregator=aggregator, gamma=gamma, name='partOf')(x)

    output.append(x)

    



    #axioms
    
    parts_of_whole, wholes_of_part = get_part_whole_ontology(classes[:-1])

    parts = {}
    wholes = {}
    
    



    for k in predictions.keys():
        if k == 'bg':
            continue
        parts[k.lower()] = keras.layers.Lambda(lambda x: get_parts(x,num_rois))(predictions[k])
        wholes[k.lower()] = keras.layers.Lambda(lambda x: get_wholes(x,num_rois))(predictions[k])

    #parts of whole
    for w in parts_of_whole.keys():

        l0 = Literal_O(False)(wholes[w])
        l1 = Literal_O(False)(partOf_prediction)
        literals = [l0,l1]

        for p in parts_of_whole[w]:
            l = Literal_O(True)(parts[p])
            literals.append(l)
        x = Clause(tnorm=tnorm, aggregator=aggregator, gamma=gamma, name='parts_of_wholes_'+w)(literals)
        output.append(x)

    #wholes of parts
    for p in wholes_of_part.keys():
        l0 = Literal_O(False)(parts[p])
        l1 = Literal_O(False)(partOf_prediction)
        literals = [l0, l1]
        for w in wholes_of_part[p]:
            l = Literal_O(True)(wholes[w])
            literals.append(l)
        x = Clause(tnorm=tnorm, aggregator=aggregator, gamma=gamma, name='wholes_of_parts_' + p)(literals)
        output.append(x)
    #disjoint of classes
    count = 0
    for t1 in classes:
        for t in classes:
            if t < t1:
                l1 = Literal_O(False)(predictions[t])
                l2 = Literal_O(False)(predictions[t1])
                x = Clause(tnorm = tnorm, aggregator = aggregator, gamma = gamma, name ='disjoint_{}_{}'.format(t,t1))([l1,l2])
                count+=1
                output.append(x)
    print(count)

    #at least one class
    at_least_literals = []
    for t in classes:
        l = Literal_O(True)(predictions[t])
        at_least_literals.append(l)
    x = Clause(tnorm = tnorm, aggregator = aggregator, gamma = gamma,name = 'at_least_one_class')(at_least_literals)
    output.append(x)







    out_ltn = keras.layers.Concatenate(axis=1)(output)
    out_ltn = keras.layers.Lambda(lambda x: keras.backend.expand_dims(x, 0))(out_ltn)
    #out_ltn = keras.layers.Lambda(lambda x: tf.Print(x,[x,x.shape],"ks"))(out_ltn)
    return [out_regr,out_ltn]

def classifierEvaluate(base_layers, input_rois, num_rois, nb_classes ,activation,trainable=False):

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

    out_class = TimeDistributed(Dense(nb_classes, activation=activation, kernel_initializer='zero'), name='dense_class_{}'.format(nb_classes))(out)
    # note: no regression target for bg class
    out_regr = TimeDistributed(Dense(4 * (nb_classes-1), activation='linear', kernel_initializer='zero'), name='dense_regress_{}'.format(nb_classes))(out)

    output = []
    for i in range(nb_classes):
        x = ltn.Predicate(num_features=nb_classes, k=6, i=i)(out_class)
        output.append(x)

        # partOF
    '''

    x = bb_creation(nb_classes, num_rois)([out_class, input_rois, base_layers])
    x = Pair(num_rois)(x)
    partOf = ltn.Predicate(num_features=(nb_classes + 5) * 2, k=6, i=nb_classes + 1)
    out_part_of = partOf(x)
    out_part_of = keras.layers.Lambda(lambda x: keras.backend.reshape(out_part_of,(1,num_rois*num_rois)))(out_part_of)
    '''
    out_ltn = keras.layers.Concatenate(axis=1)(output)
    out_ltn = keras.layers.Lambda(lambda x: keras.backend.expand_dims(x, 0))(out_ltn)


    return [out_regr,out_ltn]

def classifierPartOF(base_layers, input_rois, num_rois, nb_classes ,activation,trainable=False):
    if K.backend() == 'tensorflow':
        pooling_regions = 14
        input_shape = (num_rois,14,14,1024)
    elif K.backend() == 'theano':
        pooling_regions = 7
        input_shape = (num_rois,1024,7,7)

    out_roi_pool = RoiPoolingConv(pooling_regions, num_rois)([base_layers, input_rois])
    out = classifier_layers(out_roi_pool, input_shape=input_shape, trainable=True)

    out = TimeDistributed(Flatten())(out)

    out_class = TimeDistributed(Dense(nb_classes, activation=activation, kernel_initializer='zero'), name='dense_class_{}'.format(nb_classes))(out)
    # note: no regression target for bg class
    out_regr = TimeDistributed(Dense(4 * (nb_classes-1), activation='linear', kernel_initializer='zero'), name='dense_regress_{}'.format(nb_classes))(out)




    # predicate Dog -> partOf
    # predicate Cat ->
    # ir1
    # predicate Dog ->
    # predicate Cat ->
    # ir2 ->

    return [out_regr,out_class]

def partOf(input_part,nb_classes,num_rois):
    out_part_of = ltn.Predicate(num_features=(nb_classes + 5) * 2, k=6, i=nb_classes + 1)(input_part)
    out_part_of = keras.layers.Lambda(lambda x: keras.backend.reshape(out_part_of, (1, num_rois * num_rois)))(out_part_of)
    return out_part_of


def prova(base_layers, input_rois, num_rois, nb_classes ,activation,trainable=False):
    # compile times on theano tend to be very high, so we use smaller ROI pooling regions to workaround

    if K.backend() == 'tensorflow':
        pooling_regions = 14
        input_shape = (num_rois, 14, 14, 1024)
    elif K.backend() == 'theano':
        pooling_regions = 7
        input_shape = (num_rois, 1024, 7, 7)

    out_roi_pool = RoiPoolingConv(pooling_regions, num_rois)([base_layers, input_rois])
    out = classifier_layers(out_roi_pool, input_shape=input_shape, trainable=True)

    out = TimeDistributed(Flatten())(out)

    out_class = TimeDistributed(Dense(nb_classes, activation=activation, kernel_initializer='zero'),
                                name='dense_class_{}'.format(nb_classes))(out)
    # note: no regression target for bg class
    out_regr = TimeDistributed(Dense(4 * (nb_classes - 1), activation='linear', kernel_initializer='zero'),
                               name='dense_regress_{}'.format(nb_classes))(out)
    output = []
    predictions = {}

    # partOF

    y = bb_creation(nb_classes, num_rois)([out_class, input_rois, base_layers])
    x = Pair(num_rois)(y)
    out_part_of = ltn.Predicate(num_features=(nb_classes + 5) * 2, k=6, i=nb_classes + 1)(x)
    out_part_of = keras.layers.Lambda(lambda x: keras.backend.reshape(out_part_of, (1, num_rois//2 * num_rois//2)))(out_part_of)

    return [x,y,out_part_of]


def classifierEvaluatePartOF(base_layers, input_rois, num_rois, nb_classes ,activation,trainable=False):

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

    out_class = TimeDistributed(Dense(nb_classes, activation=activation, kernel_initializer='zero'), name='dense_class_{}'.format(nb_classes))(out)



    # partOF


    x = bb_creation(nb_classes, num_rois)([out_class, input_rois, base_layers])
    x = Pair(num_rois)(x)
    partOf = ltn.Predicate(num_features=(nb_classes + 5) * 2, k=6, i=nb_classes + 1)
    out_part_of = partOf(x)
    out_part_of = keras.layers.Lambda(lambda x: keras.backend.reshape(out_part_of,(1,num_rois*num_rois)))(out_part_of)



    return [out_part_of]