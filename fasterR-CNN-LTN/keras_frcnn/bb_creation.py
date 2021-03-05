import tensorflow as tf
from keras.layers import Layer,Activation
from keras import backend as K
from tensorflow.contrib.specs.python.specs_ops import Cx

class bb_creation(Layer):

    def __init__(self, classes, num_rois, **kwargs):
        super(bb_creation, self).__init__(**kwargs)
        self.classes = classes
        self.num_rois = num_rois

    def build(self, input_shape):
        super(bb_creation, self).build(input_shape)

    def compute_output_shape(self, inputShape):

        return [(inputShape[0][1], inputShape[0][1], inputShape[0][2] + 4) for _ in range(self.classes - 1)]

    def call(self, x, mask=None):

        assert (len(x) == 3)

        out_class = x[0]
        rois = x[1]
        b = x[2]

        # Richiedo come input la feature map per poter ottener la lunghezza e la larghezza di questa e normalizzare (x1,x2,y1,y2) per avere un input conforme con quello richiesto dalla LTN
        if K.image_dim_ordering() == 'th':
            H = K.cast(tf.shape(b)[2],'float32')
            W = K.cast(tf.shape(b)[3],'float32')
        else:
            H = K.cast(tf.shape(b)[1],'float32')
            W = K.cast(tf.shape(b)[2],'float32')
        p = []
        for roi_idx in range(self.num_rois):
            x = rois[0, roi_idx, 0]
            y = rois[0, roi_idx, 1]
            w = rois[0, roi_idx, 2]
            h = rois[0, roi_idx, 3]
            re = tf.stack([x / W, y / H, (x + w) / W, (y + h) / H])

            re = tf.expand_dims(re, axis=0)
            re = K.cast(re, 'float32')
            p.append(tf.concat((out_class[:,roi_idx,:], re), 1))
        h = tf.concat(p,axis = 0)
        h = tf.expand_dims(h, axis=0)
        return h
