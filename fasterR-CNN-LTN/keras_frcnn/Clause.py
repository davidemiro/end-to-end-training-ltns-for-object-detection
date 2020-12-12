import tensorflow as tf
from keras.layers import Layer,Activation


class Clause(Layer):
    
    def __init__(self,tnorm,aggregator,num_class, **kwargs):
        super(Clause,self).__init__(**kwargs)
        self.tnorm =tnorm
        self.aggregator = aggregator
        self.num_class = num_class
           

       
    def build(self, input_shape):
        super(Clause,self).build(input_shape)
    def compute_output_shape(self,inputShape):
        return [(1,1)]
    def call(self,input, mask=None):
        x = input[0]
        y = input[1]
        m = input[2]
        y = tf.boolean_mask(y,m)
        
        
        
        #y = tf.Print(y,[y],"y_{}".format(self.num_class))
        
        x = tf.math.multiply(y,x) + tf.math.multiply((1 - y),(1-x))
        
        #x = tf.Print(x,[x],"x_{}".format(self.num_class))
        
        pos = tf.constant([0.9 for i in range(16)])
        neg = tf.constant([0.1 for i in range(16)])
        
        weight = tf.math.multiply(y,pos) + tf.math.multiply((1 - y),neg)
        
        #weight = tf.Print(weight,[weight],"weight_{}".format(self.num_class))
        

        result = x

        
        if self.tnorm == "product":
            result = 1.0-tf.reduce_prod(1.0-x,1,keep_dims=True)
        if self.tnorm =="yager2":
            result = tf.minimum(1.0,tf.sqrt(tf.reduce_sum(tf.square(x),1, keep_dims=True)))
        if self.tnorm =="luk":
            result = tf.minimum(1.0,tf.reduce_sum(x,1, keep_dims=True))
        if self.tnorm == "goedel":
            result = tf.reduce_max(x,1,keep_dims=True,name=label)
        if self.aggregator == "product":
            return tf.reduce_prod(result,keep_dims=True)
        if self.aggregator == "mean":
            return tf.reduce_mean(result,keep_dims=True,name=label)
        if self.aggregator == "gmean":
            return tf.exp(tf.mul(tf.reduce_sum(tf.log(result), keep_dims=True),tf.inv(tf.to_float(tf.size(result)))),name=label)
        if self.aggregator == "hmean":
            h = tf.div(tf.reduce_sum(weight),tf.reduce_sum(tf.div(weight,result),keep_dims=True))
           # h = tf.Print(h,[h],"h_{}".format(self.num_class))
            return h
        if self.aggregator == "min":
            return tf.reduce_min(result, keep_dims=True)
class Literal_Clause(Layer):
    def __init__(self,num_class, **kwargs):
        super(Literal_Clause,self).__init__(**kwargs)
        self.num_class = num_class
  

       
    def build(self, input_shape):
        super(Literal_Clause,self).build(input_shape)
    def compute_output_shape(self,inputShape):
        return [(1,1)]
    def call(self, input, mask=None):
        x = input[0]
        y = input[1]
        y = tf.reshape(y,[32,1])
        
        pt = tf.math.multiply(y,x) + tf.math.multiply((1 - y),(1-x))
        pos = tf.math.multiply(tf.math.pow((1 - x),2),tf.math.log(x))
        neg = tf.math.multiply(tf.math.pow(x,2),tf.math.log(1 - x))
        weight = tf.math.multiply(pos,y) + tf.math.multiply(neg,(1 - y))
        #t - norm luk
        pt = tf.minimum(1.0,tf.reduce_sum(pt,1, keep_dims=True))
        
        #weighted hmean
        h = tf.div(tf.reduce_sum(weight),tf.reduce_sum(tf.div(weight,pt),keep_dims=True))
        

        
        return h

   
        
        
        
        
        
   

        

