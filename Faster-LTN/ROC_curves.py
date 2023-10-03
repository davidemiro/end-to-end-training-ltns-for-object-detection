import pickle
import numpy as np
from sklearn.metrics import roc_curve,auc,precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
import os,shutil




models =['PASCAL','focal_logsum_neptune','focal_logsum_neptune_alpha-2','focal_logsum_neptune_alpha_prof']
models_name =['FRCNN','Arch13 (focal loss)','Arch14 (focal loss con alpha)','Arch15 (focal loss con alpha per pos e neg)']

models = ['PASCAL','focal_logsum_neptune','focal_logsum_neptune_alpha_prof','focal_logsum_bg_no_alpha','focal_logsum_bg_alpha']
models_name =['FasterRCNN','FasterRCNN-LTN','FasterRCNN-LTN with alpha','FasterRCNN-LTN with bg','FasterRCNN-LTN with alpha and bg']
aps = [0.71,0.58,0.57,0.72,0.63]
models = ['parts','focal_logsum_bg_no_alpha_PASCAL_parts_9','model_focal_logsum_bg_PASCAL_parts_knowledge_300.hdf5','model_focal_logsum_bg_PASCAL_parts_knowledge_partOf_best_293.hdf5']
models_name =['FasterRCNN','Faster-LTN','Faster-LTN knowledge','Faster-LTN knowledge partOf']
#models = ['parts','focal_logsum_bg_no_alpha_PASCAL_parts_9']
#models_name =['FasterRCNN','Faster-LTN']
aps = [0.72,0.71,0.73,0.72]

def loadTP(name):
    T_file = open('/Users/davidemiro/Desktop/measures/T_{}.pkl'.format(name),'rb')
    P_file = open('/Users/davidemiro/Desktop/measures/P_{}.pkl'.format(name),'rb')

    T = pickle.load(T_file)
    P =pickle.load(P_file)
    return T,P



def plot_roc_curve(models,T,P,label,path,aps):



    aucs = []

    plt.title('Precision-Recall curves')
    for string,t,p,AP in zip(models,T,P,aps):
        y_test = []
        y_score = []

        for l in label:
            y_test = y_test + t[l]
            y_score = y_score + p[l]
        p,r,th = precision_recall_curve(y_test, y_score)


        plt.plot(r,p, label="{}, mAP={}".format(string,str(AP)))



    plt.legend(loc = 0)
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.savefig(os.path.join(path,'AP_PASCAL_parts_kn.png'))













T, P = loadTP('parts')

labels = list(T.keys())
T = []
P = []
for m in models:
    t, p = loadTP(m)
    T.append(t)
    P.append(p)

plot_roc_curve(models_name, T, P, labels,'/Users/davidemiro/Desktop/measures',aps)
plt.show()



