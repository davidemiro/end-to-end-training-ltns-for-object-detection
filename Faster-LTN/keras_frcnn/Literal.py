import tensorflow as tf
from keras.layers import Layer, Activation



# Implementation of the Literal module as Keras Layer
# See https://arxiv.org/pdf/1705.08968.pdf for more details
class Literal(Layer):

    def __init__(self,name,batch_size,**kwargs):
        super(Literal, self).__init__(**kwargs)
        self.name = name
        self.batch_size = batch_size


    def build(self, input_shape):
        super(Literal, self).build(input_shape)



    def call(self, input, mask=None):


        x = input[0]
        y = input[1]




        x = tf.reshape(x, (self.batch_size, 1))
        y = tf.reshape(y, (self.batch_size, 1))
        pt = tf.math.multiply(y, x) + tf.math.multiply((1 - y), (1 - x))

        return pt
