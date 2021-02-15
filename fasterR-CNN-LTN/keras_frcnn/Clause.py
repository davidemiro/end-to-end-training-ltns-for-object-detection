import tensorflow as tf
from keras.layers import Layer,Activation





class Clause(Layer):
    
    def __init__(self,tnorm,aggregator,num_class,gamma,alpha_pos=None,alpha_neg=None,**kwargs):
        super(Clause,self).__init__(**kwargs)
        self.tnorm = tnorm
        self.aggregator = aggregator
        self.num_class = num_class
        self.gamma = gamma
        self.alpha_pos = alpha_pos
        self.alpha_neg = alpha_neg

           

       
    def build(self, input_shape):
        super(Clause,self).build(input_shape)
    def compute_output_shape(self,inputShape):
        return [(1,1)]
    def call(self,input, mask=None):

        print('Predicate {}'.format(self.num_class))


        pt = input[0]
        y  = input[1]


        if self.tnorm == "product":
            result = 1.0-tf.reduce_prod(1.0-pt,1,keep_dims=True)
        if self.tnorm =="yager2":
            result = tf.minimum(1.0,tf.sqrt(tf.reduce_sum(tf.square(pt),1, keep_dims=True)))
        if self.tnorm =="luk":
            pt = tf.minimum(1.0,tf.reduce_sum(pt,1, keep_dims=True))
        if self.tnorm == "goedel":
            result = tf.reduce_max(pt,1,keep_dims=True)


        if self.aggregator == "product":
            return tf.reduce_prod(result,keep_dims=True)
        if self.aggregator == "mean":
            return tf.reduce_mean(result,keep_dims=True)
        if self.aggregator == "gmean":
            return tf.exp(tf.mul(tf.reduce_sum(tf.log(result), keep_dims=True),tf.inv(tf.to_float(tf.size(result)))))
        if self.aggregator == "hmean":
            h = tf.div(tf.to_float(tf.size(result)), tf.reduce_sum(tf.reciprocal(result), keep_dims=True))
            return h
        if self.aggregator == "min":
            return tf.reduce_min(result, keep_dims=True)
        if self.aggregator == "logsum":
            h = tf.negative(tf.reduce_sum(tf.log(pt), keep_dims=True,name="Clause_{}".format(self.num_class)))
            return h
        if self.aggregator == "focal_loss_logsum":

            fl = tf.math.multiply(tf.math.pow((1 - pt), self.gamma), tf.math.log(pt))
            fl = tf.negative(fl)
            h = tf.reduce_sum(fl, keep_dims=True,name="Clause_{}".format(self.num_class))
            return h

        
        
        
        
        
   

        

