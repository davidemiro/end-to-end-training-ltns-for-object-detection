import tensorflow as tf
from keras.layers import Layer,Activation

    
class bb_creation(Layer):
    
    def __init__(self,classes, **kwargs):
        super(bb_creation,self).__init__(**kwargs)
        self.classes = classes   

       
    def build(self, input_shape):
        super(bb_creation,self).build(input_shape)
    def compute_output_shape(self,inputShape):
    
        return [(inputShape[0][1],inputShape[0][1],inputShape[0][2] + 4) for _ in range(self.classes-1)]
    def call(self, x, mask=None):
        

        assert(len(x) == 2)

        out_regr = x[0]
        out_class = x[1]
        
        out_regrs = tf.split(out_regr,self.classes - 1 ,axis=2)
        tensors = []
        for regr in out_regrs:
            tensors.append(tf.concat([out_class,regr],2))
        return tensors
'''
import tensorflow as tf
from keras.layers import Layer,Activation
import keras.backend as K

def applyRegr(x,y,w,h,tx,ty,tw,th,W,X):
    
    tx = tf.math.divide(tx,C.classifier_regr_std[0])
    ty = tf.math.divide(ty,C.classifier_regr_std[1])
    tw = tf.math.divide(tw,C.classifier_regr_std[2])
    th = tf.math.divide(th,C.classifier_regr_std[3])

    cx = x + w/2.
    cy = y + h/2.
    cx1 = tx * w + cx
    cy1 = ty * h + cy
    w1 = tf.math.exp(tw) * w
    h1 = tf.math.exp(th) * h
    x1 = cx1 - w1/2.
    y1 = cy1 - h1/2.
    x1 = tf.math.round(x1)
    y1 = tf.math.round(y1)
    w1 = tf.math.round(w1)
    h1 = tf.math.round(h1)
    
    x1 = x1 * 16
    y1 = y1 * 16
    w1 = w1 * 16
    h1 = h1 * 16
    
    x1 = x1 / W
    y1 = y1 / H
    x2 = (x1 + w1) / W
    y2 = (y1 + h1) / H
    
    return x1, y1, x2, y2

    
class bb_creation(Layer):
    
    def __init__(self,classes,num_rois, **kwargs):
        super(bb_creation,self).__init__(**kwargs)
        self.classes = classes
        self.num_rois = num_rois   

       
    def build(self, input_shape):
        super(bb_creation,self).build(input_shape)
    def compute_output_shape(self,inputShape):
    
        return [(inputShape[0][1],inputShape[0][1],inputShape[0][2] + 4) for _ in range(self.classes-1)]
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
        return tensors
'''