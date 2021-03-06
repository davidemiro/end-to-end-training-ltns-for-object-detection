import tensorflow as tf
from keras.layers import Layer,Activation
from keras_frcnn.bb_creation import  bb_creation
import csv



def containment_ratios_between_two_bbxes(bb1, bb2):
    bb1_area = (bb1[-2] - bb1[-4]) * (bb1[-1] - bb1[-3])
    bb2_area = (bb2[-2] - bb2[-4]) * (bb2[-1] - bb2[-3])
    w_intersec = tf.math.maximum(0.0,tf.math.minimum(bb1[-2], bb2[-2]) - tf.math.maximum(bb1[-4], bb2[-4]))
    h_intersec = tf.math.maximum(0.0,tf.math.minimum(bb1[-1], bb2[-1]) - tf.math.maximum(bb1[-3], bb2[-3]))
    bb_area_intersection = w_intersec * h_intersec
    return [bb_area_intersection/bb1_area, bb_area_intersection/bb2_area]

class Pair(Layer):
    def __init__(self, batch_size,**kwargs):
        super(Pair, self).__init__(**kwargs)
        self.batch_size = batch_size
    def build(self, input_shape):
        super(Pair, self).build(input_shape)
    def call(self, inputs, **kwargs):
        outputs = []
        for i in range(self.batch_size//2):
            for j in range(self.batch_size//2):
                cts = containment_ratios_between_two_bbxes(inputs[0, i, :], inputs[0, j, :])
                x = tf.concat([inputs[0,i,:],inputs[0,j,:],cts],axis=0)
                x = tf.expand_dims(tf.expand_dims(x, axis=0), axis=0)
                outputs.append(x)
        return tf.concat(outputs,axis=1)

def get_part_whole_ontology(selected_types):
    selected_types = [s.lower() for s in selected_types]
    with open('keras_frcnn/pascalPartOntology.csv') as f:
        ontologyReader = csv.reader(f)
        parts_of_whole = {}
        wholes_of_part = {}
        for row in ontologyReader:
            parts_of_whole[row[0]] = row[1:]
            for t in row[1:]:
                if t in wholes_of_part:
                    wholes_of_part[t].append(row[0])
                else:
                    wholes_of_part[t] = [row[0]]
        for whole in parts_of_whole:
            wholes_of_part[whole] = []
        for part in wholes_of_part:
            if part not in parts_of_whole:
                parts_of_whole[part] = []
    selected_parts_of_whole = {}
    selected_wholes_of_part = {}
    for t in selected_types:
        selected_parts_of_whole[t] = [p for p in parts_of_whole[t] if p in selected_types]
        selected_wholes_of_part[t] = [w for w in wholes_of_part[t] if w in selected_types]
    return selected_parts_of_whole, selected_wholes_of_part
class Clause(Layer):
    
    def __init__(self,tnorm,aggregator,name,gamma,**kwargs):
        super(Clause,self).__init__(**kwargs)
        self.tnorm = tnorm
        self.aggregator = aggregator
        self.gamma = gamma
        self.name = name


           

       
    def build(self, input_shape):
        super(Clause,self).build(input_shape)
    def compute_output_shape(self,inputShape):
        return [(1,1)]
    def call(self,input, mask=None):


        pt = tf.concat(input,1)


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
            h = tf.div(tf.to_float(tf.size(pt)), tf.reduce_sum(tf.reciprocal(pt), keep_dims=True))
            return h
        if self.aggregator == "min":
            return tf.reduce_min(result, keep_dims=True)
        if self.aggregator == "logsum":
            h = tf.negative(tf.reduce_sum(tf.log(pt), keep_dims=True,name="Clause_"+self.name))
            return h
        if self.aggregator == "focal_loss_logsum":

            fl = tf.math.multiply(tf.math.pow((1 - pt), self.gamma), tf.math.log(pt))
            fl = tf.negative(fl)
            h = tf.reduce_sum(fl, keep_dims=True,name="Clause_"+self.name)
            return h



        
        
        
        
        
   

        

