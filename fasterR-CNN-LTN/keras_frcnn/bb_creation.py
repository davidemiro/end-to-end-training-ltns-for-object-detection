import tensorflow as tf
from keras.layers import Layer,Activation
from keras import backend as K
from tensorflow.contrib.specs.python.specs_ops import Cx

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

class bb_creation_v3(Layer):
    
    def __init__(self,classes,num_rois,std_x,std_y,std_w,std_h,**kwargs):
        super(bb_creation_v3,self).__init__(**kwargs)
        self.classes = classes
        self.num_rois = num_rois  
        self.std_x = std_x
        self.std_y = std_y
        self.std_w = std_w
        self.std_h = std_h
         

       
    def build(self, input_shape):
        super(bb_creation_v3,self).build(input_shape)
    def compute_output_shape(self,inputShape):
    
        return [(inputShape[0][1],inputShape[0][1],inputShape[0][2] + 4) for _ in range(self.classes-1)]
    def call(self, x, mask=None):
        



        out_regr = x[0]
        out_class = x[1]
        rois = x[2]
        b = x[3]
        mask = x[4:]

        if K.image_dim_ordering() == 'th':
            H = tf.shape(b)[2]
            W = tf.shape(b)[3]
        else:
            H = tf.shape(b)[1]
            W = tf.shape(b)[2]  
        
        
        
        
        out_regrs = tf.split(out_regr,self.classes - 1 ,axis=2)
        tensors = []
   
        
        i = 0
        for regr in out_regrs:
            p = []
            for roi_idx in range(self.num_rois):
                x = rois[0, roi_idx, 0]
                y = rois[0, roi_idx, 1]
                w = rois[0, roi_idx, 2]
                h = rois[0, roi_idx, 3]
                tx = regr[0, roi_idx, 0]/self.std_x
                ty = regr[0, roi_idx, 1]/self.std_y
                tw = regr[0, roi_idx, 2]/self.std_w
                th = regr[0, roi_idx, 3]/self.std_h
                
                cx = x + w/2
                cy = y + h/2
                cx1 = tx*w + cx
                cy1 = ty*h + cy
                w1 = tf.math.exp(tw)*w
                h1 = tf.math.exp(th)*h
                
                x1 = cx1 - w1/2
                y1 = cy1 - h1/2
                

                
                gx = K.cast(tf.math.round(x1),'int32')
                gy = K.cast(tf.math.round(y1),'int32')
                gw = K.cast(tf.math.round(w1),'int32')
                gh = K.cast(tf.math.round(h1),'int32')
                

                
                
                p.append(tf.stack([gx/W,gy/H,(gx + gw)/W,(gy + gh)/H]))
            
            
            re = tf.stack(p)
            
            
            re = tf.expand_dims(re,axis=0)
            re = K.cast(re,'float32')
            #re = tf.Print(re,[re],"bb")
            input = tf.concat([out_class,re],2)
            
            input = tf.boolean_mask(input,tf.reshape(mask[i],(1,32)),axis=0)
            #input_neg = tf.boolean_mask(input,f.reshape(mask[i + 1],(1,32)),axis=0)
            i += 1
            #tensors.append(input_pos)
            #tensors.append(input_neg)
            tensors.append(input)
        return tensors
    
class bb_creation_v4(Layer):
    
    def __init__(self,classes,num_rois,std_x,std_y,std_w,std_h,**kwargs):
        super(bb_creation_v4,self).__init__(**kwargs)
        self.classes = classes
        self.num_rois = num_rois  
        self.std_x = std_x
        self.std_y = std_y
        self.std_w = std_w
        self.std_h = std_h
         

       
    def build(self, input_shape):
        super(bb_creation_v4,self).build(input_shape)
    def compute_output_shape(self,inputShape):
    
        return [(inputShape[0][1],inputShape[0][1],inputShape[0][2] + 4) for _ in range((self.classes-1)*2)]
    def call(self, x, mask=None):
        



        out_regr = x[0]
        out_class = x[1]
        rois = x[2]
        b = x[3]
        mask = x[4:]

        if K.image_dim_ordering() == 'th':
            H = tf.shape(b)[2]
            W = tf.shape(b)[3]
        else:
            H = tf.shape(b)[1]
            W = tf.shape(b)[2]  
        
        
        
        
        out_regrs = tf.split(out_regr,self.classes - 1 ,axis=2)
        tensors = []
   
        
        i = 0
        for regr in out_regrs:
            p = []
            for roi_idx in range(self.num_rois):
                x = rois[0, roi_idx, 0]
                y = rois[0, roi_idx, 1]
                w = rois[0, roi_idx, 2]
                h = rois[0, roi_idx, 3]
                tx = regr[0, roi_idx, 0]/self.std_x
                ty = regr[0, roi_idx, 1]/self.std_y
                tw = regr[0, roi_idx, 2]/self.std_w
                th = regr[0, roi_idx, 3]/self.std_h
                
                cx = x + w/2
                cy = y + h/2
                cx1 = tx*w + cx
                cy1 = ty*h + cy
                w1 = tf.math.exp(tw)*w
                h1 = tf.math.exp(th)*h
                
                x1 = cx1 - w1/2
                y1 = cy1 - h1/2
                

                
                gx = K.cast(tf.math.round(x1),'int32')
                gy = K.cast(tf.math.round(y1),'int32')
                gw = K.cast(tf.math.round(w1),'int32')
                gh = K.cast(tf.math.round(h1),'int32')
                

                
                
                p.append(tf.stack([gx/W,gy/H,(gx + gw)/W,(gy + gh)/H]))
            
            
            re = tf.stack(p)
            
            
            re = tf.expand_dims(re,axis=0)
            re = K.cast(re,'float32')
            #re = tf.Print(re,[re],"bb")
            input = tf.concat([out_class,re],2)
            
            input_pos = tf.boolean_mask(input,tf.reshape(mask[i],(1,32)),axis=0)
            input_neg = tf.boolean_mask(input,tf.reshape(mask[i + 1],(1,32)),axis=0)
            i += 1
            tensors.append(input_pos)
            tensors.append(input_neg)

        return tensors

class bb_creation_evaluate(Layer):
    
    def __init__(self,classes,num_rois,std_x,std_y,std_w,std_h,**kwargs):
        super(bb_creation_evaluate,self).__init__(**kwargs)
        self.classes = classes
        self.num_rois = num_rois  
        self.std_x = std_x
        self.std_y = std_y
        self.std_w = std_w
        self.std_h = std_h
         

       
    def build(self, input_shape):
        super(bb_creation_evaluate,self).build(input_shape)
    def compute_output_shape(self,inputShape):
    
        return [(inputShape[0][1],inputShape[0][1],inputShape[0][2] + 4) for _ in range(self.classes-1)]
    def call(self, x, mask=None):
        



        out_regr = x[0]
        out_class = x[1]
        rois = x[2]
        b = x[3]
        mask = x[4:]

        if K.image_dim_ordering() == 'th':
            H = tf.shape(b)[2]
            W = tf.shape(b)[3]
        else:
            H = tf.shape(b)[1]
            W = tf.shape(b)[2]  
        
        
        
        
        out_regrs = tf.split(out_regr,self.classes - 1 ,axis=2)
        tensors = []
   
        
        i = 0
        for regr in out_regrs:
            p = []
            for roi_idx in range(self.num_rois):
                x = rois[0, roi_idx, 0]
                y = rois[0, roi_idx, 1]
                w = rois[0, roi_idx, 2]
                h = rois[0, roi_idx, 3]
                tx = regr[0, roi_idx, 0]/self.std_x
                ty = regr[0, roi_idx, 1]/self.std_y
                tw = regr[0, roi_idx, 2]/self.std_w
                th = regr[0, roi_idx, 3]/self.std_h
                
                cx = x + w/2
                cy = y + h/2
                cx1 = tx*w + cx
                cy1 = ty*h + cy
                w1 = tf.math.exp(tw)*w
                h1 = tf.math.exp(th)*h
                
                x1 = cx1 - w1/2
                y1 = cy1 - h1/2
                

                
                gx = K.cast(tf.math.round(x1),'int32')
                gy = K.cast(tf.math.round(y1),'int32')
                gw = K.cast(tf.math.round(w1),'int32')
                gh = K.cast(tf.math.round(h1),'int32')
                

                
                
                p.append(tf.stack([gx/W,gy/H,(gx + gw)/W,(gy + gh)/H]))
            
            
            re = tf.stack(p)
            
            
            re = tf.expand_dims(re,axis=0)
            re = K.cast(re,'float32')
            input = tf.concat([out_class,re],2)
            tensors.append(input)
        return tensors