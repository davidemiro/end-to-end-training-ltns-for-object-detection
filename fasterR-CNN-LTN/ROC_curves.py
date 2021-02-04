import pickle
import numpy as np
from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt



models =['PASCAL','focal_logsum_neptune-2','focal_logsum_neptune_alpha-2','focal_logsum_neptune_alpha_prof']
def loadTP(name):
    T_file = open('/Users/davidemiro/Downloads/T_{}.pkl'.format(name),'rb')
    P_file = open('/Users/davidemiro/Downloads/P_{}.pkl'.format(name),'rb')

    T = pickle.load(T_file)
    P =pickle.load(P_file)
    return T,P

def plot_roc_curve(models,T,P,label):



    aucs = []


    for string,t,p in zip(models,T,P):


        y_test = t[label]
        y_test_roc = np.array([([0, 1] if y else [1, 0]) for y in y_test])

        y_score = p[label]

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)


        plt.plot(fpr, tpr, label="{}, auc={}".format(string,str(roc_auc)))

    plt.legend(loc = 0)
    plt.savefig('/Users/davidemiro/Desktop/ROC_curves_FRCNN-LTN/{}.png'.format(label))
    plt.show()







T = []
P = []

for m in models:
    t,p = loadTP(m)
    T.append(t)
    P.append(p)

labels =list(P[0].keys())
for l in labels:
    plot_roc_curve(models,T,P,l)


