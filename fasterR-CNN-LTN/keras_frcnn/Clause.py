import tensorflow as tf
from keras.layers import Layer,Activation





class Clause(Layer):
    
    def __init__(self,tnorm,aggregator,num_class,alpha,gamma,**kwargs):
        super(Clause,self).__init__(**kwargs)
        self.tnorm = tnorm
        self.aggregator = aggregator
        self.num_class = num_class
        self.alpha = alpha
        self.gamma = gamma

           

       
    def build(self, input_shape):
        super(Clause,self).build(input_shape)
    def compute_output_shape(self,inputShape):
        return [(1,1)]
    def call(self,input, mask=None):

        if len(input) ==3:
            x = input[0]
            y = input[1]
            m = input[2]
        else:
            x = input[0]
            y = input[1]

        #literal
        #x = tf.Print(x, [x,tf.shape(x)], "Prediction_{}".format(self.num_class))
        y = tf.Print(y, [y,tf.shape(y)], "Labels_{}".format(self.num_class),summarize=20000)
        x = tf.reshape(x,(32,1))
        y = tf.reshape(y,(32,1))
        pt = tf.math.multiply(y, x) + tf.math.multiply((1 - y), (1 - x))
       # x_ = tf.Print(x_,[x_,tf.shape(x_)],"Litteral_{}".format(self.num_class))
        pt = tf.reshape(pt,(32,1))
        #pt = tf.Print(pt, [pt, tf.shape(pt)], "Literal_{}".format(self.num_class))
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
           # result =tf.Print(result, [result,tf.shape(result)], "result_{}".format(self.num_class))
            h = tf.negative(tf.reduce_sum(tf.log(pt), keep_dims=True,name="Clause_{}".format(self.num_class)))
            #print(self.num_class)
            #h = tf.Print(h, [h,tf.shape(h)], "Clause_{}".format(self.num_class))
            return h
        if self.aggregator == "focal_los_logsum":
            h = tf.math.multiply(tf.math.pow((1 - pt), self.gamma), tf.math.log(pt))
            h = tf.Print(h, [h], "h_{}".format(self.num_class),summarize=20000)
            fl = tf.negative(self.alpha*h)
            fl = tf.Print(fl, [fl], "fl_{}".format(self.num_class),summarize=20000)
            h = tf.reduce_sum(fl, keep_dims=True,name="Clause_{}".format(self.num_class))
           # h = tf.Print(h, [h], "h_{}".format(self.num_class))
            return h

        
        
        
        
        
   

        

