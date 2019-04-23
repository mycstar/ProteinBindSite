import pandas as pd
import os
import numpy as np
from datetime import datetime

from sklearn.metrics import classification_report, roc_auc_score, roc_curve, make_scorer, confusion_matrix, \
    recall_score, precision_score, f1_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, GroupKFold
from sklearn.metrics import auc, precision_recall_curve
from sklearn.utils import shuffle
from sklearn import tree, svm, naive_bayes, neighbors
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelBinarizer
from scipy import interp
import matplotlib.pyplot as plt

group_num = 7
split_size = 5


def typicalSampling(group, group_num):
    name = group.name
    return group.sample(n=group_num)


clfs = {'svm': svm.SVC(probability=True),
        'decision_tree': tree.DecisionTreeClassifier(),
        'naive_gaussian': naive_bayes.GaussianNB(),
        'naive_mul': naive_bayes.MultinomialNB(),
        'K_neighbor': neighbors.KNeighborsClassifier(),
        'bagging_knn': BaggingClassifier(neighbors.KNeighborsClassifier(), max_samples=0.5, max_features=0.5),
        'bagging_tree': BaggingClassifier(tree.DecisionTreeClassifier(), max_samples=0.5, max_features=0.5),
        'random_forest': RandomForestClassifier(n_estimators=50),
        'adaboost': AdaBoostClassifier(n_estimators=50),
        'gradient_boost': GradientBoostingClassifier(n_estimators=50, learning_rate=1.0, max_depth=1, random_state=0)
        }


def measure_predict(y_true, y_predict):
    lb = LabelBinarizer()
    bin_true = pd.Series(lb.fit_transform(y_true)[:, 0])
    bin_predict = pd.Series(lb.fit_transform(y_predict)[:, 0])

    print("confusion matrix: \n{}\n".format(
        confusion_matrix(bin_true, bin_predict)))
    print("recall_score: {}\n".format(
        recall_score(bin_true, bin_predict)))
    print("precision_score: {}\n".format(
        precision_score(bin_true, bin_predict)))
    print("f1_score : {}\n".format(
        f1_score(bin_true, bin_predict)))


def try_different_method(clf, train_x, train_y, test_x, test_y):
    clf.fit(train_x, train_y.ravel())
    score = clf.score(test_x, test_y.ravel())
    print('the score is :', score)
    print("pos count:", test_y.sum())
    pred = clf.predict(test_x)
    measure_predict(test_y, pred)


def try_different_method_probas(clf, train_x, train_y, test_x, test_y):
    clf.fit(train_x, train_y.ravel())
    probas_ = clf.predict_proba(test_x)
    return test_y, probas_


def plot_cv_roc(clf_name, ret_list):
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    i = 0
    for ret in ret_list:
        test_y, probas_ = ret[0], ret[1]
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(test_y, probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                 label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

        i += 1

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(clf_name + ' Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig("../mid_result/" + clf_name + '.png')
    plt.close()
    # plt.show()


df = pd.read_csv('../mid_result/mid_result00.csv')

pos_dt = df[df["label"] == 1]
neg_dt = df[df["label"] == 0]

groups = pd.Series(df["pdbid"].tolist()).unique()

print("total different protein is:", len(groups))

neg_target_dt = neg_dt.groupby(
    'pdbid', group_keys=False
).apply(typicalSampling, group_num)

kf = GroupKFold(n_splits=split_size)

groupData = pd.concat([pos_dt, neg_target_dt])

families = np.array(groupData["pdbid"].tolist())
labels = np.array(groupData["label"].tolist())
seqs = groupData.iloc[:, 6:15].values

max_value = seqs.max(axis=0)
min_value = np.min(seqs, axis=0)

ret_pred = {}
print("train pdb: ", pd.Series(families).unique())
print("total data is:", len(seqs))
start_time = datetime.now()

for fold, (train, test) in enumerate(kf.split(seqs, labels, families)):
    print("Fold :", fold, " starting")
    train_x = seqs[train]
    train_y = labels[train]
    family_train_1 = families[train]
    print("train pdb: ", pd.Series(family_train_1).unique())
    print("train data is:", len(train_y))

    test_x = seqs[test].astype(np.float64)
    test_y = labels[test]
    family_test_1 = families[test]
    print("test family: ", pd.Series(family_test_1).unique())
    print("test data is:", len(test_y))

    train_x, train_y, family_train_1 = shuffle(train_x, train_y, family_train_1, random_state=1233)

    for clf_key in clfs.keys():
        # print('the classifier is :', clf_key)
        clf = clfs[clf_key]
        test_y, probas_ = try_different_method_probas(clf, train_x, train_y, test_x, test_y)
        if clf_key in ret_pred:
            ret = ret_pred[clf_key]
            ret.append([test_y, probas_])
        else:
            ret_pred[clf_key] = [[test_y, probas_]]

    print('haha')

for key, value in ret_pred.items():
    plot_cv_roc(key, value)

end_time = datetime.now()
print("The total Duration: {}".format(end_time - start_time))

print('haha')
