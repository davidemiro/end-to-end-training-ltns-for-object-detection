import pickle
import numpy as np
from sklearn.metrics import roc_curve,auc,precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score




models =['PASCAL','focal_logsum_neptune-2','focal_logsum_neptune_alpha-2','focal_logsum_neptune_alpha_prof']
models_name =['FRCNN','Arch13 (focal loss)','Arch14 (focal loss con alpha)','Arch15 (focal loss con alpha per pos e neg)']

#models = ['parts','focal_logsum_pos_weights_parts']
#models_name =['FRCNN','Arch15 (focal loss con alpha per pos e neg)']
def loadTP(name):
    T_file = open('/Users/davidemiro/Downloads/T_{}.pkl'.format(name),'rb')
    P_file = open('/Users/davidemiro/Downloads/P_{}.pkl'.format(name),'rb')

    T = pickle.load(T_file)
    P =pickle.load(P_file)
    return T,P

def plot_roc_curve(models,T,P,label):



    aucs = []

    plt.title('ROC curves {} class'.format(label))
    for string,t,p in zip(models,T,P):


        y_test = t[label]
        y_test_roc = np.array([([0, 1] if y else [1, 0]) for y in y_test])

        y_score = p[label]

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()


        p,r,th = precision_recall_curve(y_test, y_score)

        if (string in ['FRCNN','Arch13 (focal loss)'] and label == 'bird'):
            print(string)
            print('Precision')
            print(p)
            print('Recall')
            print(r)
            print('Threshold')
            print(th)
        AP = average_precision_score(y_test, y_score)
        plt.plot(r,p, label="{}, ap={}".format(string,str(AP)))



    plt.legend(loc = 0)
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.savefig('/Users/davidemiro/Desktop/AP_curves_pascal_part/ROC_{}.png'.format(label))
    plt.show()







T = []
P = []

for m in models:
    t,p = loadTP(m)
    T.append(t)
    P.append(p)

labels =list(P[0].keys())
for l in labels:
    plot_roc_curve(models_name,T,P,l)


