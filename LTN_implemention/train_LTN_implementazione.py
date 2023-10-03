import numpy as np
from keras import backend as K
from keras.optimizers import Adam, SGD, RMSprop
from keras.layers import Input,Flatten,Concatenate
from keras.models import Model

import neptune
from keras.utils import generic_utils
from keras.models import load_model
import ltn
from keras.layers.merge import concatenate
import random
import json
from pascalpart import *
import keras
import tensorflow as tf
import keras
import random


if True:
    train_data, pairs_of_train_data, types_of_train_data, partOf_of_pairs_of_train_data = get_data_unofficial("train")
    idxs_of_positive_examples_of_types = {}
    idxs_of_negative_examples_of_types = {}
    types_of_train_data = np.array([t.decode('UTF-8').lower() for t in types_of_train_data])

    
    types_of_train_data =  np.where(types_of_train_data == 'artifact_win','artifact_wing',types_of_train_data)
    types_of_train_data =  np.where(types_of_train_data == 'license_plat','license_plate',types_of_train_data)
    types_of_train_data = np.array([np.bytes_(t) for t in types_of_train_data])
    for type in selected_types:
        idxs_of_positive_examples_of_types[type] = np.where(types_of_train_data == type)[0]
        idxs_of_negative_examples_of_types[type] = np.where(types_of_train_data != type)[0]

    idxs_of_positive_examples_of_partOf = np.where(partOf_of_pairs_of_train_data==1)[0]
    idxs_of_negative_examples_of_partOf = np.where(partOf_of_pairs_of_train_data == 0)[0] 


    existing_types = [t for t in selected_types if idxs_of_positive_examples_of_types[t].size > 0]
    
else:
    train_data, pairs_of_train_data, types_of_train_data, partOf_of_pairs_of_train_data, _, _ = get_data("train",max_rows=1000000000)

# computing positive and negative exampls for types and partof

    idxs_of_positive_examples_of_types = {}
    idxs_of_negative_examples_of_types = {}

    for type in selected_types:
        idxs_of_positive_examples_of_types[type] = np.where(types_of_train_data == type)[0]
        idxs_of_negative_examples_of_types[type] = np.where(types_of_train_data != type)[0]

    idxs_of_positive_examples_of_partOf = np.where(partOf_of_pairs_of_train_data)[0]
    idxs_of_negative_examples_of_partOf = np.where(partOf_of_pairs_of_train_data == False)[0] 


existing_types = [t for t in selected_types if idxs_of_positive_examples_of_types[t].size > 0]

def clause_Layer(input):
    result = tf.minimum(1.0,tf.reduce_sum(input,1, keep_dims=True))
    h = tf.div(tf.to_float(tf.size(result)),tf.reduce_sum(tf.reciprocal(result),keep_dims=True))
    return h

def add_noise_to_data(noise_ratio):

    if noise_ratio > 0:
        freq_other = {}

        for t in selected_types:
            freq_other[t] = {}
            number_of_not_t = len(idxs_of_negative_examples_of_types[t])
            for t1 in selected_types:
                if t1 != t:
                    freq_other[t][t1] = np.float(len(idxs_of_positive_examples_of_types[t1]))/number_of_not_t

        noisy_data_idxs = np.random.choice(range(len(train_data)), int(len(train_data) * noise_ratio),replace=False)

        for idx in noisy_data_idxs:
            type_of_idx = types_of_train_data[idx]
            not_types_of_idx = np.setdiff1d(selected_types,type_of_idx)
            
            types_of_train_data[idx] = np.random.choice(not_types_of_idx,
                                                        p=np.array([freq_other[type_of_idx][t1] \
                                                                    for t1 in not_types_of_idx]))

        noisy_data_pairs_idxs = np.append(np.random.choice(np.where(partOf_of_pairs_of_train_data)[0],
                                                 int(partOf_of_pairs_of_train_data.sum() * noise_ratio * 0.5)),
                                          np.random.choice(np.where(np.logical_not(partOf_of_pairs_of_train_data))[0],
                                                           int(partOf_of_pairs_of_train_data.sum() * noise_ratio* 0.5)))

        for idx in noisy_data_pairs_idxs:
            partOf_of_pairs_of_train_data[idx] = not (partOf_of_pairs_of_train_data[idx])

    idxs_of_noisy_positive_examples_of_types = {}
    idxs_of_noisy_negative_examples_of_types = {}

    for type in selected_types:
        idxs_of_noisy_positive_examples_of_types[type] = np.where(types_of_train_data == type)[0]
        idxs_of_noisy_negative_examples_of_types[type] = np.where(types_of_train_data != type)[0]

    idxs_of_noisy_positive_examples_of_partOf = np.where(partOf_of_pairs_of_train_data)[0]
    idxs_of_noisy_negative_examples_of_partOf = np.where(partOf_of_pairs_of_train_data == False)[0]

    print("I have introduces the followins errors")
    for t in selected_types:
        print("wrong positive", t, len(np.setdiff1d(idxs_of_noisy_positive_examples_of_types[t],
                                                    idxs_of_positive_examples_of_types[t])))
        print("wrong negative", t, len(np.setdiff1d(idxs_of_noisy_negative_examples_of_types[t],
                                                    idxs_of_negative_examples_of_types[t])))

    print("wrong positive partof", len(np.setdiff1d(idxs_of_noisy_positive_examples_of_partOf,
                                                    idxs_of_positive_examples_of_partOf)))
    print("wrong negative partof", len(np.setdiff1d(idxs_of_noisy_negative_examples_of_partOf,
                                                    idxs_of_negative_examples_of_partOf)))

    return idxs_of_noisy_positive_examples_of_types, \
           idxs_of_noisy_negative_examples_of_types, \
           idxs_of_noisy_positive_examples_of_partOf, \
           idxs_of_noisy_negative_examples_of_partOf,

def get_inputs(idxs_of_pos_ex_of_types,
                  idxs_of_neg_ex_of_types,
                  idxs_of_pos_ex_of_partOf,
                  idxs_of_neg_ex_of_partOf,
                  pairs_data,
                  with_constraints,batch_size,partOF):
    print("selecting new training data")
    feed_dict = []


    # positive and negative examples for types
    if True:
        for t in existing_types:
            feed_dict.append(
                train_data[np.random.choice(idxs_of_pos_ex_of_types[t],
                                        batch_size)][:,:])
    
        for t in existing_types:
            feed_dict.append(
                train_data[np.random.choice(idxs_of_neg_ex_of_types[t],
                                        batch_size)][:, :])
    else:
        for t in existing_types:
            feed_dict.append(
                train_data[np.random.choice(idxs_of_pos_ex_of_types[t],
                                        batch_size)][:,1:])
    
        for t in existing_types:
            feed_dict.append(
                train_data[np.random.choice(idxs_of_neg_ex_of_types[t],
                                        batch_size)][:, 1:])

    
    if partOF:
        feed_dict.append(pairs_of_train_data[np.random.choice(idxs_of_pos_ex_of_partOf,batch_size)])
        feed_dict.append(pairs_of_train_data[np.random.choice(idxs_of_neg_ex_of_partOf,batch_size)])
    
    return feed_dict

def logictensornetwork(input,k,nb_classes,partOF):
    
    objects = []
    not_objects = []
    

    for i in range(nb_classes):
        j = i + nb_classes
        input_positive = input[i]
        input_negative = input[j]
        p = ltn.Predicate(num_features=nb_classes+5,k=k,i=i)
        object = p(input_positive)
        not_object = p(input_negative)
        not_object =keras.layers.Lambda(lambda o:1-o)(not_object)
        object = keras.layers.Lambda(lambda o: clause_Layer(o))(object)
        not_object = keras.layers.Lambda(lambda o: clause_Layer(o))(not_object)
        objects.append(object)
        not_objects.append(not_object)

    if partOF:
        input_positive_partOF = input[-2]
        input_negative_partOF = input[-1]
        p = ltn.Predicate((nb_classes+5)*2,k,'partOF')
        partOF = p(input_positive_partOF)
        not_partOF = p(input_negative_partOF)
        not_partOF = keras.layers.Lambda(lambda o:1-o)(not_partOF)
        partOF = [keras.layers.Lambda(lambda o: clause_Layer(o))(partOF)]
        not_partOF = [keras.layers.Lambda(lambda o: clause_Layer(o))(not_partOF)]
        h = concatenate(objects+not_objects+partOF+not_partOF)
    else:
        h = concatenate(objects+not_objects)
    return h

def train(params):
    
    
    #model definition
    num_classes = 59
    num_iterations = params['num_iterations'] 
    batch_size = params['batch_size']


    inputs = [Input(shape=(num_classes+5,)) for i in range(num_classes*2)]
    if params['partOF']:
        inputs+=[Input(shape=((num_classes+5) * 2,)) for i in range(2)]

  
    logic = logictensornetwork(inputs,params['k'],num_classes,params['partOF'])
    model_ltn = Model(inputs=inputs,outputs=logic)
    if params['optimizer'] == 'RMSprop':
        optimizer_ltn =RMSprop(lr=params['learning_rate'])
    else:
        optimizer_ltn =None
    model_ltn.compile(optimizer=optimizer_ltn, loss=ltn.ltn_loss(model_ltn,params['smooth_factor']))
    
    #training
    
    # add noise to train data
    
    idxs_of_noisy_positive_examples_of_types, \
    idxs_of_noisy_negative_examples_of_types, \
    idxs_of_noisy_positive_examples_of_partOf, \
    idxs_of_noisy_negative_examples_of_partOf = add_noise_to_data(params['noise_ratio'])

    
    y = np.ones([params['batch_size'],118])
    
    for i in range(num_iterations):
     
        if i % params['frequency_of_feed_dict_generation'] == 0:    
            x = get_inputs(idxs_of_noisy_positive_examples_of_types, 
                       idxs_of_noisy_negative_examples_of_types, 
                       idxs_of_noisy_positive_examples_of_partOf, 
                       idxs_of_noisy_negative_examples_of_partOf, 
                       pairs_of_train_data,
                       params['constraints'],params['batch_size'],params['partOF']) 
        
        pred = model_ltn.predict_on_batch(x)
        sat_lev = np.shape(pred)[1]/np.sum(np.reciprocal(pred))
        if sat_lev < params['saturation_limit']:
            loss = model_ltn.train_on_batch(x,y)
            neptune.log_metric('loss',loss)
        print(loss)
    print("saving")
    model_ltn.save_weights('LTN_implementazione_unofficial_features_{}_{}.h5'.format(params['noise_ratio'],params['constraints']))

neptune.init('davidemiro/sandbox', api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiYjI1NGI5OWItYjJjMC00MTY0LThkYTctOTdmMjYyMWZkNDEyIn0=')
for nr in [0.0, 0.1, 0.2, 0.3, 0.4]:
    for wc in [False]:
        
        params ={
            'num_iterations':1000,
            'batch_size':250,
            'frequency_of_feed_dict_generation':100,
            'constraints':wc,
            'noise_ratio':nr,
            'saturation_limit':.95,
            'data':'unofficial',
            'optimizer':"RMSprop",
            'learning_rate':1e-2,
            'decay':0.9,
            'k':6,
            'partOF':True,
            'smooth_factor':1e-10,
            'default_tnorm':'luk',
            'default_aggregator': "hmean",
            'default_positive_fact_penality':1e-6,
            'default_clauses_aggregator': "hmean",
            
        }
        
        exp_name = 'LTN_implementazione_unofficial_features_classification_partOf_official_features{}_{}'.format(nr,wc)
        neptune.create_experiment(name=exp_name,
                          params=params)
        neptune.append_tag('LTN_implementation')
        neptune.append_tag('unofficial_features')
        neptune.append_tag('with_constrains='.format(wc))
        neptune.append_tag('noise='.format(nr))
        
        

        train(params=params)
