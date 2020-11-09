'''
from keras import backend as K
from keras.layers import Layer,Activation
from keras.layers.merge import concatenate
from tensorflow.contrib.layers.python.layers.regularizers import l2_regularizer
import tensorflow as tf


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

        self.up = self.add_weight(name='up', # 1 x k
                                  shape = (self.k,1),
                                  initializer='uniform',
                                  trainable=True)
         
        self.Wp = self.add_weight(name='Wp', 
                                  shape = (self.k,mn,mn), 
                                  initializer= 'uniform', 
                                  trainable = True)
        self.Vp = self.add_weight(name='Vp', 
                                  shape = (self.k,mn), 
                                  initializer= 'uniform', 
                                  trainable = True)
        self.bp = self.add_weight(name='bp', 
                                  shape = (1,self.k), 
                                  initializer= 'uniform', 
                                  trainable = True)
  
        
        super(Predicate, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        
        #import pdb;pdb.set_trace()
        X = x
        #X = tf.Print(X,[tf.shape(X)])
        XW = tf.matmul(tf.tile(tf.expand_dims(X, 0), [self.k, 1, 1]), self.Wp)
        XWX = tf.squeeze(tf.matmul(tf.expand_dims(X, 1), tf.transpose(XW, [1, 2, 0])))
        XV = tf.matmul(X, tf.transpose(self.Vp))
        gX = tf.matmul(tf.tanh(XWX + XV + self.bp),self.up)
        h = tf.sigmoid(gX)
        
        return h

        
        
        x = K.reshape(x,(self.num_features,1))
        WT_Wp= K.dot(K.transpose(x),self.Wp)
        vT_Wp_v = K.dot(WT_Wp,x)
        Vp_v = K.dot(self.Vp,x)
        sum = vT_Wp_v +Vp_v + self.bp
        tanh = K.tanh(sum)
        out = K.sigmoid(K.dot(K.transpose(self.up),tanh))
    
        
        
        #return h
def loss_ltn(model,params):
    
    def loss(y_true,y_pred):
        return smooth(model.get_weights(),params['smooth_factor'])-tf.div(tf.to_float(tf.size(y_pred)), tf.reduce_sum(tf.reciprocal(y_pred), keep_dims=True))
    return loss
def smooth(parameters,default_smooth_factor):
    
    norm_of_omega = tf.reduce_sum(tf.expand_dims(tf.concat(
                     [tf.expand_dims(tf.reduce_sum(tf.square(par)),0) for par in parameters],axis=0),1))
    return tf.multiply(default_smooth_factor,norm_of_omega)
'''

from keras import backend as K
from keras.layers import Layer,Activation
from keras.layers.merge import concatenate
from tensorflow.contrib.layers.python.layers.regularizers import l2_regularizer
import tensorflow as tf


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

        self.up = self.add_weight(name='up', # 1 x k
                                  shape = (self.k,1),
                                  initializer='random_normal',
                                  trainable=True)
         
        self.Wp = self.add_weight(name='Wp', 
                                  shape = (self.k,mn,mn), 
                                  initializer= 'random_normal', 
                                  trainable = True)
        self.Vp = self.add_weight(name='Vp', 
                                  shape = (self.k,mn), 
                                  initializer= 'random_normal', 
                                  trainable = True)
        self.bp = self.add_weight(name='bp', 
                                  shape = (1,self.k), 
                                  initializer= 'random_normal', 
                                  trainable = True)
      

        
        super(Predicate, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        

        X = x
        XW = tf.matmul(tf.tile(tf.expand_dims(X, 0), [self.k, 1, 1]), self.Wp)
        XWX = tf.squeeze(tf.matmul(tf.expand_dims(X, 1), tf.transpose(XW, [1, 2, 0])))
        XV = tf.matmul(X, tf.transpose(self.Vp))
        gX = tf.matmul(tf.tanh(XWX + XV + self.bp),self.up)
        h = tf.sigmoid(gX)
        
        return h

def ltn_loss(model,smooth_factor):
    def loss(y_true,y_pred):
        return smooth(model.get_weights(),smooth_factor)-tf.div(tf.to_float(tf.size(y_pred)), tf.reduce_sum(tf.reciprocal(y_pred), keep_dims=True))
    return loss

def smooth(parameters,default_smooth_factor):
    
    norm_of_omega = tf.reduce_sum(tf.expand_dims(tf.concat(
                     [tf.expand_dims(tf.reduce_sum(tf.square(par)),0) for par in parameters],axis=0),1))
    return tf.multiply(default_smooth_factor,norm_of_omega)