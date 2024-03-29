from keras import backend as K
from keras.layers import Layer,Activation
from keras.layers.merge import concatenate
from tensorflow.contrib.layers.python.layers.regularizers import l2_regularizer
import tensorflow as tf



# Implementation of the Predicate module of Logic Tensor Networks following https://arxiv.org/pdf/1705.08968.pdf

class Predicate(Layer):
    def __init__(self,num_features,k,i, **kwargs):
        super(Predicate,self).__init__(**kwargs)
        self.output_dim = 1
        self.num_features = num_features 
        self.k = k
        self.name ='predicate_{}'.format(i) 
        

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        mn = self.num_features
        label = self.name
        layers = self.k



        self.up = self.add_weight(name='up'+self.name, # 1 x k
                                  shape = (self.k,1),
                                  initializer='ones',
                                  trainable=True)
         
        self.Wp = self.add_weight(name='Wp'+self.name, 
                                  shape = (self.k,mn,mn), 
                                  initializer= 'random_normal', 
                                  trainable = True)
        self.Vp = self.add_weight(name='Vp'+self.name, 
                                  shape = (self.k,mn), 
                                  initializer= 'random_normal', 
                                  trainable = True)
        self.bp = self.add_weight(name='bp'+self.name, 
                                  shape = (1,self.k), 
                                  initializer= 'ones', 
                                  trainable = True)
        self.bp = self.bp*-1

      

        
        super(Predicate, self).build(input_shape)

    def call(self, x):

        X = tf.squeeze(x)
        XW = tf.matmul(tf.tile(tf.expand_dims(X, 0), [self.k, 1, 1]), self.Wp)
        XWX = tf.squeeze(tf.matmul(tf.expand_dims(X, 1), tf.transpose(XW, [1, 2, 0])))
        XV = tf.matmul(X, tf.transpose(self.Vp))
        gX = tf.matmul(tf.tanh(XWX + XV + self.bp),self.up)
        h = tf.sigmoid(gX)

        return h

           



def ltn_loss(type,weight):
    def ltn_loss(y_true,y_pred):
        if type == 'hmean':
            return -tf.div(tf.reduce_sum(y_true), tf.reduce_sum(tf.div(y_true,y_pred), keep_dims=True)+tf.constant(1e-15))
        elif type == 'sum':
            return tf.reduce_sum(y_pred, keep_dims=True)
        else:
            return None
    return ltn_loss




def smooth(parameters,default_smooth_factor):
    
    norm_of_omega = tf.reduce_sum(tf.expand_dims(tf.concat(
                     [tf.expand_dims(tf.reduce_sum(tf.square(par)),0) for par in parameters],axis=0),1))
    return tf.multiply(default_smooth_factor,norm_of_omega)
