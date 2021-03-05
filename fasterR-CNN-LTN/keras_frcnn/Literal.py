import tensorflow as tf
from keras.layers import Layer, Activation


class Literal_O(Layer):
    def __init__(self,polarity):
        super(Literal_O, self).__init__()
        self.polarity = polarity
    def built(self, input_shape):
        super(Literal_O, self).built(input_shape)
    def call(self, input, mask=None):
        if self.polarity:
            return input
        else:
            return 1 - input

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



        # literal
        #x = tf.Print(x, [x,tf.shape(x)], "Prediction_{}".format(self.num_class))
        #y = tf.Print(y, [y,tf.shape(y)], "Labels_{}".format(self.num_class),summarize=20000)
        x = tf.reshape(x, (self.batch_size, 1))
        y = tf.reshape(y, (self.batch_size, 1))
        pt = tf.math.multiply(y, x) + tf.math.multiply((1 - y), (1 - x))
        # x_ = tf.Print(x_,[x_,tf.shape(x_)],"Litteral_{}".format(self.num_class))
        #pt = tf.Print(pt, [pt], "Literal_{}".format(self.num_class))
        return pt
