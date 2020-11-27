import tensorflow as tf
from keras.layers import Layer,Activation
from keras import backend as K

#Questo file contiene l 'implementazione della class bb_creation che fornisce un layer per concatenare l' output del softmax FRCNN con l' output di regressione sempre del FRCNN per generare i tensori di input della LTN
#Nei vari allenamenti questa classe è stata aggiornata,quindi ci sono anche la v1 e v2  
class bb_creation(Layer):
    
    def __init__(self,classes, **kwargs):
        super(bb_creation,self).__init__(**kwargs)
        self.classes = classes   

       
    def build(self, input_shape):
        super(bb_creation,self).build(input_shape)
    def compute_output_shape(self,inputShape):
    
        return [(inputShape[0][1],inputShape[0][1],inputShape[0][2] + 4) for _ in range(self.classes-1)]
    def call(self, x, mask=None):
        
        #In questa versione il layer riceve come input i due output out_class e out_regr della FRCNN
        #La out_regr è un tensore di dimensione (1,num_rois,num_classes * 4).Ogni classe ha il suo "quartetto di regressione".
        
        assert(len(x) == 2)

        out_regr = x[0]
        out_class = x[1]
        #Faccio lo split del tensore originale generando una lista di tensori (1,num_rois,4) di classes - 1(background) elementi in modo tale da dividere le varie quadruple associate ad ogni classe.
        
        out_regrs = tf.split(out_regr,self.classes - 1 ,axis=2)
        tensors = []
        
        for regr in out_regrs:
            tensors.append(tf.concat([out_class,regr],2))
        return tensors
class bb_creation_v1(Layer):
    
    def __init__(self,classes,num_rois, **kwargs):
        super(bb_creation_v1,self).__init__(**kwargs)
        self.classes = classes
        self.num_rois = num_rois   

       
    def build(self, input_shape):
        super(bb_creation_v1,self).build(input_shape)
    def compute_output_shape(self,inputShape):
    
        return [(inputShape[0][1],inputShape[0][1],inputShape[0][2] + 4) for _ in range(self.classes-1)]
    def call(self, x, mask=None):
        

        assert(len(x) == 4)

        out_regr = x[0]
        out_class = x[1]
        rois = x[2]
        b = x[3]
        
        #Richiedo come input la feature map per poter ottener la lunghezza e la larghezza di questa e normalizzare (x1,x2,y1,y2) per avere un input conforme con quello richiesto dalla LTN
        if K.image_dim_ordering() == 'th':
            H = tf.shape(b)[2]
            W = tf.shape(b)[3]
        else:
            H = tf.shape(b)[1]
            W = tf.shape(b)[2]  
        
        
        
        
        out_regrs = tf.split(out_regr,self.classes - 1 ,axis=2)
        tensors = []
   
        
        
        #In questo caso combino out_regr con le ROIs,seguendo il paper Fast RCNN
        for regr in out_regrs:
            p = []
            for roi_idx in range(self.num_rois):
                x = rois[0, roi_idx, 0]
                y = rois[0, roi_idx, 1]
                w = rois[0, roi_idx, 2]
                h = rois[0, roi_idx, 3]
                tx = regr[0, roi_idx, 0]
                ty = regr[0, roi_idx, 1]
                tw = regr[0, roi_idx, 2]
                th = regr[0, roi_idx, 3]
                
                cx = w*tx
                cy = h*ty
                cw = tf.math.exp(tw)
                ch = tf.math.exp(th)
                
                gx = cx + x
                gy = cy + y
                gw = w * cw
                gh = h * ch
                
                
                gx = K.cast(tf.math.round(gx),'int32')
                gy = K.cast(tf.math.round(gy),'int32')
                gw = K.cast(tf.math.round(gw),'int32')
                gh = K.cast(tf.math.round(gh),'int32')
                
                
                p.append(tf.stack([gx/W,gy/H,(gx + gw)/W,(gy + gh)/H]))
            
  
            re = tf.stack(p)
            
            
            re = tf.expand_dims(re,axis=0)
            re = K.cast(re,'float32')
            
         
       
            
            tensors.append(tf.concat([out_class,re],2))
        return tensors

class bb_creation_v2(Layer):
    
    def __init__(self,classes,num_rois, **kwargs):
        super(bb_creation_v2,self).__init__(**kwargs)
        self.classes = classes
        self.num_rois = num_rois   

       
    def build(self, input_shape):
        super(bb_creation_v2,self).build(input_shape)
    def compute_output_shape(self,inputShape):
    
        return [(inputShape[0][1],inputShape[0][1],inputShape[0][2] + 4) for _ in range(self.classes)]
    def call(self, x, mask=None):
        

        assert(len(x) == 4)

        out_regr = x[0]
        out_class = x[1]
        rois = x[2]
        b = x[3]
    
        if K.image_dim_ordering() == 'th':
            H = tf.shape(b)[2]
            W = tf.shape(b)[3]
        else:
            H = tf.shape(b)[1]
            W = tf.shape(b)[2]  
        
        
        
        
        out_regrs = tf.split(out_regr,self.classes - 1 ,axis=2)
        tensors = []
   
        
        
  
        for regr in out_regrs:
            p = []
            for roi_idx in range(self.num_rois):
                x = rois[0, roi_idx, 0]
                y = rois[0, roi_idx, 1]
                w = rois[0, roi_idx, 2]
                h = rois[0, roi_idx, 3]
                tx = regr[0, roi_idx, 0]
                ty = regr[0, roi_idx, 1]
                tw = regr[0, roi_idx, 2]
                th = regr[0, roi_idx, 3]
                
                cx = w*tx
                cy = h*ty
                cw = tf.math.exp(tw)
                ch = tf.math.exp(th)
                
                gx = cx + x
                gy = cy + y
                gw = w * cw
                gh = h * ch
                
                
                gx = K.cast(tf.math.round(gx),'int32')
                gy = K.cast(tf.math.round(gy),'int32')
                gw = K.cast(tf.math.round(gw),'int32')
                gh = K.cast(tf.math.round(gh),'int32')
                
                
                p.append(tf.stack([gx/W,gy/H,(gx + gw)/W,(gy + gh)/H]))
            
  
            re = tf.stack(p)
            
            
            re = tf.expand_dims(re,axis=0)
            re = K.cast(re,'float32')
            
         
       
            
            tensors.append(tf.concat([out_class,re],2))
        
        p = []
        #in questa versione aggiungo anche il tensore di input del predicate background
        for roi_idx in range(self.num_rois):
            x = rois[0, roi_idx, 0]
            y = rois[0, roi_idx, 1]
            w = rois[0, roi_idx, 2]
            h = rois[0, roi_idx, 3]
            
            x = K.cast(x,'int32')
            y = K.cast(y,'int32')
            w = K.cast(w,'int32')
            h = K.cast(h,'int32')
            
            p.append(tf.stack([x/W,y/H,(x + w)/W,(y + h)/H]))
        
        reb = tf.stack(p)
        reb = tf.expand_dims(reb,axis=0)
        reb = K.cast(reb,'float32')
        
        tensors.append(tf.concat([out_class,reb],2))
        
        return tensors