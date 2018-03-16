from sklearn.kernel_ridge import KernelRidge
from numpy.linalg import inv
import numpy as np
import pandas as pd


def calculate_risk(real_x, reco_x, predict_correct, sigma):
    raw_risk = np.linalg.norm(real_x - reco_x) / (2 * sigma * sigma)
    if predict_correct:
        risk = np.exp(-raw_risk)
    else:
        risk = np.exp(raw_risk)
    return risk


if __name__ == "__main__":

    tra_csv = pd.read_csv('data/nursery-ssl10-10-1tra.csv')
    trs_csv = pd.read_csv('data/nursery-ssl10-10-1trs.csv')
    tst_csv = pd.read_csv('data/nursery-ssl10-10-1tst.csv')

    tra_len, trs_len, tst_len = len(tra_csv.index), len(trs_csv.index), len(tst_csv.index)

    merge_csv = pd.concat([tra_csv, trs_csv, tst_csv])

    merge_csv_dummy_y = pd.get_dummies(merge_csv['class'])
    NUM_CLASSES = len(merge_csv_dummy_y.columns)

    merge_csv.drop(['class'], axis=1, inplace=True)
    merge_csv_dummy_x = pd.get_dummies(merge_csv)

    tra_dummy_y = merge_csv_dummy_y[0:tra_len]
    trs_dummy_y = merge_csv_dummy_y[tra_len:(tra_len + trs_len)]
    tst_dummy_y = merge_csv_dummy_y[(tra_len + trs_len):]

    tra_dummy_x = merge_csv_dummy_x[0:tra_len]
    trs_dummy_x = merge_csv_dummy_x[tra_len:(tra_len + trs_len)]
    tst_dummy_x = merge_csv_dummy_x[(tra_len + trs_len):]

    tra_dummy_y_labeled = tra_dummy_y.loc[tra_dummy_y['unlabeled'] != 1]
    tra_dummy_x_labeled = tra_dummy_x.loc[tra_dummy_x.index.isin(tra_dummy_y_labeled.index)]

    tra_dummy_y_unlabeled = tra_dummy_y.loc[tra_dummy_y['unlabeled'] == 1]
    tra_dummy_x_unlabeled = tra_dummy_x.loc[tra_dummy_x.index.isin(tra_dummy_y_unlabeled.index)]

    del tra_csv
    del trs_csv
    del tst_csv
    del merge_csv
    del tra_dummy_y
    del tra_dummy_x

    y = tra_dummy_y_labeled.as_matrix()
    X = tra_dummy_x_labeled.as_matrix()

    print("Now fitting primal RLS classifier")
    # PRIMAL FORM RLS
    clf = KernelRidge(alpha=0.05)
    clf.fit(X, y)

    gamma = 0.05
    K = clf._get_kernel(X)
    Id = np.identity(len(X))

    # Equation 4
    alpha_star = np.matmul(inv(K + gamma * (Id)), y)

    print("Now training dual CRC classifier")
    # DUAL FORM CRC
    XT = np.transpose(X)
    XTX = np.matmul(XT, X)

    Id = np.identity(len(XTX))
    alpha_k = np.matmul(np.matmul(inv(XTX + gamma * Id), XT), y)

    tra_dummy_x_unlabeled['risk'] = 999
    tra_dummy_y_unlabeled[:] = 0
    for i in range(len(trs_dummy_x)):

        # Xrs = trs_dummy_x.as_matrix()[i]
        # yrs = trs_dummy_y.as_matrix()[i]
        Xrs = tra_dummy_x_unlabeled.as_matrix()[i]
        yrs = tra_dummy_y_unlabeled.as_matrix()[i]

        prediction = clf.predict(np.reshape(Xrs, (1, len(Xrs))))
        predicted_class = np.argmax(prediction[0])

        # reconstruct back X instance of the predicted y
        reco_x = np.matmul(alpha_k, prediction[0])
        real_x = Xrs

        # predict y base on reconstructed X
        prediction_reco = clf.predict(np.reshape(reco_x, (1, len(Xrs))))
        predicted_class_reco = np.argmax(prediction_reco[0])

        # yXpredicted should be == yXreconstructed
        predict_correct = (predicted_class == predicted_class_reco)
        calc_risk = calculate_risk(real_x, reco_x, predict_correct, 0.05)
        tra_dummy_x_unlabeled.loc[i]['risk'] = calc_risk
        print("predicted class: " + str(predicted_class))
        print("predicted reconstructed class: " + str(predicted_class_reco))
        print("real class: " + str(np.argmax(yrs)))
        print("Risk: " + str(calc_risk))
        # set predicted class of unlabeled if calc_risk < threshold
        tra_dummy_y_unlabeled.loc[i][y.columns[predicted_class]] = 1
        input("===============Enter to continue===============")


######################################
# PRIMAL FORM RLS
# n_samples, n_features = 10, 5
# rng = np.random.RandomState(0)
# y = rng.uniform(low=0, high=1, size=[n_samples, 3])
# y = np.where(y > 0.5, 1, 0)
# X = rng.uniform(low=0, high=1, size=[n_samples, n_features])
#
# clf = KernelRidge(alpha=0.05)
# clf.fit(X, y)
#
# gamma = 0.05
# K = clf._get_kernel(X)
# Id = np.identity(len(X))
#
# # Equation 4
# alpha_star = np.matmul(inv(K + gamma * (Id)), y)
#
#
# # DUAL FORM CRC
# XT = np.transpose(X)
# XTX = np.matmul(XT, X)
#
# Id = np.identity(len(XTX))
# alpha_k = np.matmul(np.matmul(inv(XTX + gamma * Id), XT), y)
#
# prediction = np.matmul(alpha_k, res)
# print(prediction)
