import pickle
import numpy as np
from sklearn.metrics import roc_curve,auc,precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score




models =['PASCAL','focal_logsum_neptune-2','focal_logsum_neptune_alpha-2','focal_logsum_neptune_alpha_prof']
models_name =['FRCNN','Arch13 (focal loss)','Arch14 (focal loss con alpha)','Arch15 (focal loss con alpha per pos e neg)']

models = ['PASCAL','focal_logsum_neptune-2','focal_logsum_bg_no_alpha']
models_name =['FRCNN','Arch13 (focal loss no alpha)','Arch13 (focal loss no alpha) with bg']
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


        AP = average_precision_score(y_test, y_score)
        plt.plot(r,p, label="{}, ap={}".format(string,str(AP)))



    plt.legend(loc = 0)
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.savefig('/Users/davidemiro/Desktop/AP_curves_comparison/Precision_Recall_{}.png'.format(label))
    plt.show()







T = []
P = []

for m in models:
    t,p = loadTP(m)
    print(m)
    print('\n')
    all_aps = []
    for key in t.keys():
        ap = average_precision_score(t[key], p[key])
        print('{} AP: {}'.format(key, ap))
        all_aps.append(ap)
    print('mAP = {}'.format(np.mean(np.array(all_aps))))
    T.append(t)
    P.append(p)

labels =list(P[0].keys())
for l in labels:
    plot_roc_curve(models_name,T,P,l)


