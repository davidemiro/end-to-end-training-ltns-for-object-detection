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

      

        
        super(Predicate, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        
        
        
        #(1,batch,features) -> (batch,features)
        #x = tf.Print(x,[x],"x:")
        X = tf.squeeze(x)
        #X = tf.Print(X,[X],"X:")
        XW = tf.matmul(tf.tile(tf.expand_dims(X, 0), [self.k, 1, 1]), self.Wp)
#        XW = tf.Print(XW,[XW],"XW:")
        XWX = tf.squeeze(tf.matmul(tf.expand_dims(X, 1), tf.transpose(XW, [1, 2, 0])))
 #       XWX = tf.Print(XWX,[XWX],"XWX:")
        XV = tf.matmul(X, tf.transpose(self.Vp))
  #      XV = tf.Print(XV,[XV],"XV:")
        gX = tf.matmul(tf.tanh(XWX + XV + self.bp),self.up)
   #     gX = tf.Print(gX,[gX],"gX:")
        h = tf.sigmoid(gX)
        #h = tf.Print(h,[h],"h{}:".format(self.name))
        return h
class Clause(Layer):
    def __init__(self, **kwargs):
        super(Clause,self).__init__(**kwargs)
        self.name ='Clause'
        
    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        super(Clause, self).build(input_shape)  # Be sure to call this at the end
    def call(self, x):
        #(1,batch,features) -> (batch,features)

        inputs = x[0]
        A = x[1]
        B = x[2]
        inputs = tf.math.subtract(A, inputs)
        inputs = tf.matmul(inputs,B)
        result = tf.minimum(1.0,tf.reduce_sum(inputs,1, keep_dims=True))
        h = tf.div(tf.to_float(tf.size(result)),tf.reduce_sum(tf.reciprocal(result),keep_dims=True))
        return h
           


def ltn_loss(y_true,y_pred):
    return -tf.div(tf.to_float(tf.size(y_pred)), tf.reduce_sum(tf.reciprocal(y_pred), keep_dims=True))
        #return -tf.div(tf.to_float(tf.size(y_pred)), tf.reduce_sum(tf.reciprocal(y_pred), keep_dims=True))


def smooth(parameters,default_smooth_factor):
    
    norm_of_omega = tf.reduce_sum(tf.expand_dims(tf.concat(
                     [tf.expand_dims(tf.reduce_sum(tf.square(par)),0) for par in parameters],axis=0),1))
    return tf.multiply(default_smooth_factor,norm_of_omega)