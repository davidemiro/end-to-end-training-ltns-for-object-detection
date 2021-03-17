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
#from sklearn.metrics import average_precision_score

count = 0
def get_features(pred, gt, f,features):
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

config_output_filename = options.config_filename

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


#class_mapping = {'dog': 0, 'person': 1, 'cat': 2, 'bird': 3, 'bottle': 4, 'train': 5, 'sofa': 6, 'pottedplant': 7, 'sheep': 8, 'car': 9, 'bicycle': 10, 'chair': 11, 'diningtable': 12, 'tvmonitor': 13, 'motorbike': 14, 'boat': 15, 'horse': 16, 'bus': 17, 'cow': 18, 'aeroplane': 19, 'bg': 20}
class_mapping = {'Person': 0, 'Hand': 1, 'Arm': 2, 'Neck': 3, 'Torso': 4, 'Nose': 5, 'Hair': 6, 'Mouth': 7, 'Ebrow': 8, 'Eye': 9, 'Ear': 10, 'Head': 11, 'Bottle': 12, 'Cap': 13, 'Body': 14, 'Leg': 15, 'Pottedplant': 16, 'Plant': 17, 'Pot': 18, 'Foot': 19, 'Chair': 20, 'Sheep': 21, 'Tail': 22, 'Muzzle': 23, 'Cat': 24, 'Dog': 25, 'Train': 26, 'Locomotive': 27, 'Bicycle': 28, 'Handlebar': 29, 'Chain_Wheel': 30, 'Wheel': 31, 'Motorbike': 32, 'Tvmonitor': 33, 'Screen': 34, 'Horse': 35, 'Hoof': 36, 'Car': 37, 'Window': 38, 'Bodywork': 39, 'Mirror': 40, 'License_plate': 41, 'Door': 42, 'Headlight': 43, 'Saddle': 44, 'Boat': 45, 'Diningtable': 46, 'Coach': 47, 'Aeroplane': 48, 'Stern': 49, 'Sofa': 50, 'Bird': 51, 'Beak': 52, 'Artifact_Wing': 53, 'Engine': 54, 'Bus': 55, 'Animal_Wing': 56, 'Horn': 57, 'Cow': 58}

if 'bg' not in class_mapping:
	class_mapping['bg'] = len(class_mapping)

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

# define the RPN, built on the base layers
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
rpn_layers = nn.rpn(shared_layers, num_anchors)

classifier = nn.classifier(feature_map_input, roi_input, C.num_rois, nb_classes=len(class_mapping), trainable=True)

model_rpn = Model(img_input, rpn_layers)
model_classifier_only = Model([feature_map_input, roi_input], classifier)

model_classifier = Model([feature_map_input, roi_input], classifier)

model_rpn.load_weights("/content/drive/MyDrive/Tesi_Davide_Miro-main-2/fasterR-CNN/model_rpn_original.hdf5", by_name=True)
model_classifier.load_weights("/content/drive/MyDrive/Tesi_Davide_Miro-main-2/fasterR-CNN/model_classifier_original.hdf5", by_name=True)

model_rpn.compile(optimizer='sgd', loss='mse')
model_classifier.compile(optimizer='sgd', loss='mse')

all_imgs, _, _ = get_data(options.test_path)
test_imgs = [s for s in all_imgs if s['imageset'] == options.name]
#test_imgs = [test_imgs[i] for i in range(len(test_imgs)) if i < 5]

data = []
label_data = []
label_pair = []
count = 0
c = 0
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

	# convert from (x1,y1,x2,y2) to (x,y,w,h)
	R[:, 2] -= R[:, 0]
	R[:, 3] -= R[:, 1]

	# apply the spatial pyramid pooling to the proposed regions
	bboxes = {}
	lboxes = []
	for jk in range(R.shape[0] // C.num_rois + 1):
		ROIs = np.expand_dims(R[C.num_rois * jk:C.num_rois * (jk + 1), :], axis=0)
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

		[P_cls, P_regr] = model_classifier_only.predict([F, ROIs])

		for ii in range(P_cls.shape[1]):


			(x, y, w, h) = ROIs[0, ii, :]

			cls_num = np.argmax(P_cls[0, ii, :])
			num_classes = list(class_mapping.keys())
			for gt_box in img_data['bboxes']:
				gt_class = gt_box['class']
				gt_x1 = gt_box['x1'] / fx
				gt_x2 = gt_box['x2'] / fx
				gt_y1 = gt_box['y1'] / fy
				gt_y2 = gt_box['y2'] / fy
				ious = []
				if cls_num != 59:
					try:
						(tx, ty, tw, th) = P_regr[0, ii, 4 * cls_num:4 * (cls_num + 1)]
						tx /= C.classifier_regr_std[0]
						ty /= C.classifier_regr_std[1]
						tw /= C.classifier_regr_std[2]
						th /= C.classifier_regr_std[3]
						x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
					except:
						ious.append(0)
						pass
				pred_x1 = C.rpn_stride * x
				pred_y1 = C.rpn_stride * y
				pred_x2 = C.rpn_stride * (x + w)
				pred_y2 = C.rpn_stride * (y + h)
				iou = data_generators.iou((pred_x1, pred_y1, pred_x2, pred_y2), (gt_x1, gt_y1, gt_x2, gt_y2))

				if iou > 0.5:

					b = np.concatenate((np.array([c]),P_cls[0, ii, :], np.array([pred_x1, pred_y1, pred_x2, pred_y2])))

					box = {}
					box['feature'] = b
					box['label'] = gt_box['class']
					box['partOf'] = gt_box['partOf']
					box['id'] = gt_box['id']
					box['count'] = count
					count+=1
					bboxes[gt_box['id']] = box
					lboxes.append(box)


	for v in lboxes:
		data.append(v['feature'])
		label_data.append(v['label'])
		if v['id'] != v['partOf'] and v['partOf'] in bboxes:
			label_pair.append(bboxes[v['partOf']]['count'])
		else:
			label_pair.append(-1)
	c += 1


data = np.array(data)
label_data = np.array(label_data)
label_pair = np.array(label_pair)

np.savetxt('label_partOf_{}.csv'.format(options.name),label_pair,fmt='%d',delimiter=',')
np.savetxt('data_{}.csv'.format(options.name),data,fmt='%10.15f',delimiter=',')
np.savetxt('label_{}.csv'.format(options.name),label_data,fmt='%s',delimiter=',')