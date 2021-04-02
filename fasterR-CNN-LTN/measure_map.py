import os
import cv2
import numpy as np
import sys
import pickle
from optparse import OptionParser
import time
from keras_frcnn import config
from keras import backend as K
from keras.layers import Input
from keras.models import Model
from keras_frcnn import roi_helpers
from keras_frcnn import data_generators
from sklearn.metrics import average_precision_score

def containment_ratios_between_two_bbxes(bb1, bb2):
    bb1_area = (bb1[-2] - bb1[-4]) * (bb1[-1] - bb1[-3])
    bb2_area = (bb2[-2] - bb2[-4]) * (bb2[-1] - bb2[-3])
    w_intersec = np.maximum(0.0,np.minimum(bb1[-2], bb2[-2]) - np.maximum(bb1[-4], bb2[-4]))
    h_intersec = np.maximum(0.0,np.minimum(bb1[-1], bb2[-1]) - np.maximum(bb1[-3], bb2[-3]))
    bb_area_intersection = w_intersec * h_intersec
    return [bb_area_intersection/bb1_area, bb_area_intersection/bb2_area]
def bb_creation(out_class,rois,b,num_rois):


    # Richiedo come input la feature map per poter ottener la lunghezza e la larghezza di questa e normalizzare (x1,x2,y1,y2) per avere un input conforme con quello richiesto dalla LTN
    if K.image_dim_ordering() == 'th':
        H = float(np.shape(b)[2])
        W = float(np.shape(b)[3])
    else:
        H = float(np.shape(b)[2])
        W = float(np.shape(b)[3])
    p = []
    for roi_idx in range(num_rois):
        x = rois[0, roi_idx, 0]
        y = rois[0, roi_idx, 1]
        w = rois[0, roi_idx, 2]
        h = rois[0, roi_idx, 3]
        re = np.stack([x / W, y / H, (x + w) / W, (y + h) / H])

        re = np.expand_dims(re, axis=0)
        p.append(np.concatenate((out_class[:, roi_idx, :], re), 1))
    h = np.concatenate(p, axis=0)
    h = np.expand_dims(h, axis=0)
    return h
def get_map(pred, gt, f):
    T = {}
    P = {}
    fx, fy = f

    for bbox in gt:
        bbox['bbox_matched'] = False

    pred_probs = np.array([s['prob'] for s in pred])
    box_idx_sorted_by_prob = np.argsort(pred_probs)[::-1]

    for box_idx in box_idx_sorted_by_prob:
        pred_box = pred[box_idx]
        pred_class = pred_box['class']
        pred_x1 = pred_box['x1']
        pred_x2 = pred_box['x2']
        pred_y1 = pred_box['y1']
        pred_y2 = pred_box['y2']
        pred_prob = pred_box['prob']
        if pred_class not in P:
            P[pred_class] = []
            T[pred_class] = []
        P[pred_class].append(pred_prob)
        found_match = False

        for gt_box in gt:
            gt_class = gt_box['class']
            gt_x1 = gt_box['x1'] / fx
            gt_x2 = gt_box['x2'] / fx
            gt_y1 = gt_box['y1'] / fy
            gt_y2 = gt_box['y2'] / fy
            gt_seen = gt_box['bbox_matched']
            if gt_class != pred_class:
                continue
            if gt_seen:
                continue
            iou = data_generators.iou((pred_x1, pred_y1, pred_x2, pred_y2), (gt_x1, gt_y1, gt_x2, gt_y2))
            if iou >= 0.5:
                found_match = True
                gt_box['bbox_matched'] = True
                break
            else:
                continue

        T[pred_class].append(int(found_match))

    for gt_box in gt:
        if not gt_box['bbox_matched'] and not gt_box['difficult']:
            if gt_box['class'] not in P:
                P[gt_box['class']] = []
                T[gt_box['class']] = []

            T[gt_box['class']].append(1)
            P[gt_box['class']].append(0)

    # import pdb
    # pdb.set_trace()
    return T, P


sys.setrecursionlimit(40000)

parser = OptionParser()

parser.add_option("-p", "--path", dest="test_path", help="Path to test data.")
parser.add_option("-n", "--num_rois", dest="num_rois",
                  help="Number of ROIs per iteration. Higher means more memory use.", default=32)
parser.add_option("--config_filename", dest="config_filename", help=
"Location to read the metadata related to the training (generated when training).",
                  default="config.pickle")
parser.add_option("-o", "--parser", dest="parser", help="Parser to use. One of simple or pascal_voc",
                  default="pascal_voc")
parser.add_option("--name", dest="name", help="Name to give at model")

(options, args) = parser.parse_args()

if not options.test_path:  # if filename is not given
    parser.error('Error: path to test data must be specified. Pass --path to command line')

if options.parser == 'pascal_voc':
    from keras_frcnn.pascal_voc_parser import get_data
elif options.parser == 'simple':
    from keras_frcnn.simple_parser import get_data
else:
    raise ValueError("Command line option parser must be one of 'pascal_voc' or 'simple'")
#config_output_filename = '/content/drive/MyDrive/Tesi_Davide_Miro-main/fasterR-CNN-LTN/config_focal_logsum_bg_PASCAL_parts_knowledge_partOf.pickle'
config_output_filename = 'config_'+options.name+'.pickle'
with open(config_output_filename, 'rb') as f_in:
    C = pickle.load(f_in)
# turn off any data augmentation at test time
C.use_horizontal_flips = False
C.use_vertical_flips = False
C.rot_90 = False

if C.network == 'resnet50':
    import keras_frcnn.resnet as nn
elif C.network == 'vgg':
    import keras_frcnn.vgg as nn
elif C.network == 'resnet101':
    import keras_frcnn.resnet101 as nn

img_path = options.test_path


def format_img(img, C):
    img_min_side = float(C.im_size)
    (height, width, _) = img.shape

    if width <= height:
        f = img_min_side / width
        new_height = int(f * height)
        new_width = int(img_min_side)
    else:
        f = img_min_side / height
        new_width = int(f * width)
        new_height = int(img_min_side)
    fx = width / float(new_width)
    fy = height / float(new_height)
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    img = img[:, :, (2, 1, 0)]
    img = img.astype(np.float32)
    img[:, :, 0] -= C.img_channel_mean[0]
    img[:, :, 1] -= C.img_channel_mean[1]
    img[:, :, 2] -= C.img_channel_mean[2]
    img /= C.img_scaling_factor
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img, fx, fy




class_mapping = C.class_mapping

inv_map = class_mapping



class_mapping = {v: k for k, v in class_mapping.items()}
print(class_mapping)
class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}
C.num_rois = int(options.num_rois)

if K.image_dim_ordering() == 'th':
    input_shape_img = (3, None, None)
    input_shape_features = (1024, None, None)
else:
    input_shape_img = (None, None, 3)
    input_shape_features = (None, None, 1024)

img_input = Input(shape=input_shape_img)
roi_input = Input(shape=(C.num_rois, 4))
feature_map_input = Input(shape=input_shape_features)

# define the base network (resnet here, can be VGG, Inception, etc)
shared_layers = nn.nn_base(img_input, trainable=True)
input_partOf = Input(shape=(90000, 130))
# define the RPN, built on the base layers
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
rpn_layers = nn.rpn(shared_layers, num_anchors)

roi_input_p = Input(shape=(300, 4))
classifier = nn.classifierEvaluate(feature_map_input, roi_input, C.num_rois, len(class_mapping),'linear',trainable=True)
classifier_partOf = nn.classifierPartOF(feature_map_input, roi_input_p, 300, len(class_mapping),'linear',trainable=True)
part_Of_classifier = nn.partOf(input_partOf,60,300)
model_rpn = Model(img_input, rpn_layers)
model_classifier_only = Model([feature_map_input, roi_input], classifier)

model_classifier = Model([feature_map_input, roi_input], classifier)
model_classifier_partOf = Model([feature_map_input, roi_input_p], classifier_partOf)
model_partOf = Model([input_partOf],part_Of_classifier)

model_rpn.load_weights('/Users/davidemiro/Desktop/Pesi_107/fasterR-CNN-LTN/model_focal_logsum_bg_PASCAL_parts_knowledge_partOf_best_293.hdf5'.format(options.name), by_name=True)
model_classifier.load_weights('/Users/davidemiro/Desktop/Pesi_107/fasterR-CNN-LTN/model_focal_logsum_bg_PASCAL_parts_knowledge_partOf_best_293.hdf5'.format(options.name),by_name=True)
model_classifier_partOf.load_weights('/Users/davidemiro/Desktop/Pesi_107/fasterR-CNN-LTN/model_focal_logsum_bg_PASCAL_parts_knowledge_partOf_best_293.hdf5'.format(options.name),by_name=True)
model_partOf.load_weights('/Users/davidemiro/Desktop/Pesi_107/fasterR-CNN-LTN/model_focal_logsum_bg_PASCAL_parts_knowledge_partOf_best_293.hdf5'.format(options.name),by_name=True)


model_rpn.compile(optimizer='sgd', loss='mse')
model_classifier.compile(optimizer='sgd', loss='mse')

all_imgs, _, _ = get_data(options.test_path)
test_imgs = [s for s in all_imgs if s['imageset'] == 'test']

T = {}
P = {}
T_partof = []
P_partof = []
for idx, img_data in enumerate(test_imgs):
    print('{}/{}'.format(idx, len(test_imgs)))
    st = time.time()
    filepath = img_data['filepath']

    img = cv2.imread(filepath)

    X, fx, fy = format_img(img, C)

    if K.image_dim_ordering() == 'tf':
        X = np.transpose(X, (0, 2, 3, 1))

    # get the feature maps and output from the RPN
    [Y1, Y2, F] = model_rpn.predict(X)


    R = roi_helpers.rpn_to_roi(Y1, Y2, C, K.image_dim_ordering(), overlap_thresh=0.7)
    ids = [i for i in range(R.shape[0])]
    o1_o2_p = []

    Y3, detected_part_of,pair,_ = roi_helpers.calc_iou_partOf_test(R, img_data, C, inv_map)

    if Y3 == None:
        continue

    # convert from (x1,y1,x2,y2) to (x,y,w,h)
    R[:, 2] -= R[:, 0]
    R[:, 3] -= R[:, 1]

    # apply the spatial pyramid pooling to the proposed regions
    bboxes = {}
    probs = {}
    sel_rois = {}
    sels_m = set()

    for jk in range(R.shape[0] // C.num_rois + 1):
        ROIs = np.expand_dims(R[C.num_rois * jk:C.num_rois * (jk + 1), :], axis=0)
        rois_ids = ids[C.num_rois * jk:C.num_rois * (jk + 1)]
        if ROIs.shape[1] == 0:
            break
        if jk == R.shape[0] // C.num_rois:
            # pad R
            curr_shape = ROIs.shape
            target_shape = (curr_shape[0], C.num_rois, curr_shape[2])
            ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
            ROIs_padded[:, :curr_shape[1], :] = ROIs
            ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
            ROIs = ROIs_padded
            rois_ids[curr_shape[1]:C.num_rois] = [rois_ids[0] for _ in range(C.num_rois - curr_shape[1])]

        [P_regr,P_cls] = model_classifier_only.predict([F, ROIs])

        for ii in range(P_cls.shape[1]):

            if np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
                continue

            cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]

            if cls_name not in bboxes:
                bboxes[cls_name] = []
                probs[cls_name] = []
                sel_rois[cls_name] = []

            (x, y, w, h) = ROIs[0, ii, :]

            cls_num = np.argmax(P_cls[0, ii, :])
            try:
                (tx, ty, tw, th) = P_regr[0, ii, 4 * cls_num:4 * (cls_num + 1)]
                tx /= C.classifier_regr_std[0]
                ty /= C.classifier_regr_std[1]
                tw /= C.classifier_regr_std[2]
                th /= C.classifier_regr_std[3]
                x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
            except:
                pass
            bboxes[cls_name].append([16 * x, 16 * y, 16 * (x + w), 16 * (y + h)])
            probs[cls_name].append(np.max(P_cls[0, ii, :]))
            sel_rois[cls_name].append(rois_ids[ii])

    ROIs = np.expand_dims(R, axis=0)
    _,out_class = model_classifier_partOf.predict([F, ROIs])

    inputs = bb_creation(out_class,ROIs,F,300)

    inputs_part_of = []
    for i in range(300):
        for j in range(300):
            cts = containment_ratios_between_two_bbxes(inputs[0, i, :], inputs[0, j, :])
            x = np.concatenate([inputs[0, i, :], inputs[0, j, :], cts], axis=0)
            x = np.expand_dims(np.expand_dims(x, axis=0), axis=0)
            inputs_part_of.append(x)
    inputs_part_of = np.concatenate(inputs_part_of, axis=1)

    out_part_of = model_partOf.predict([inputs_part_of])

    ii = 0
    for i in range(300):
        for j in range(300):
            o1_o2_p.append((i,j,out_part_of[0,ii],Y3[i][j]))
            ii += 1



    all_dets = []
    dets_rois = set()

    for key in bboxes:
        bbox = np.array(bboxes[key])

        new_boxes, new_probs, new_sel_rois = roi_helpers.non_max_suppression_fast_partOf(bbox, np.array(probs[key]),np.array(sel_rois[key]), overlap_thresh=0.5)
        for jk in range(new_boxes.shape[0]):
            (x1, y1, x2, y2) = new_boxes[jk, :]
            det = {'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'class': key, 'prob': new_probs[jk]}
            all_dets.append(det)
            dets_rois.add(new_sel_rois[jk])

    t_part_of = []
    p_part_of = []

    true_dets = {}

    for c in o1_o2_p:
        if Y3[c[0]][c[1]] == 1:
            t_part_of.append(c[3])
            p_part_of.append(c[2])
    '''
    for b in img_data['bboxes']:
        if b['partOf'] != b['id']:
            if '{}{}'.format(b['id'], b['partOf']) not in detected_part_of:
                t_part_of.append(1)
                p_part_of.append(0)
    '''

    T_partof.extend(t_part_of)
    P_partof.extend(p_part_of)

    print('Elapsed time = {}'.format(time.time() - st))

    ap = average_precision_score(T_partof, P_partof)
    print('PartOf AP: {}'.format(ap))

    print('Elapsed time = {}'.format(time.time() - st))
    t, p = get_map(all_dets, img_data['bboxes'], (fx, fy))
    for key in t.keys():
        if key not in T:
            T[key] = []
            P[key] = []
        T[key].extend(t[key])
        P[key].extend(p[key])
    all_aps = []
    for key in T.keys():
        ap = average_precision_score(T[key], P[key])
        print('{} AP: {}'.format(key, ap))
        all_aps.append(ap)
    print('mAP = {}'.format(np.mean(np.array(all_aps))))
T['partOf'] = T_partof
P['partOf'] = P_partof
with open('T_' + options.name + '.pkl', 'wb') as f:
    pickle.dump(T, f)
with open('P_' + options.name + '.pkl', 'wb') as f:
    pickle.dump(P, f)