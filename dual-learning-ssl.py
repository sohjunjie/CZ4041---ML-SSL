import os

from sklearn.kernel_ridge import KernelRidge
from numpy.linalg import inv
import numpy as np
import pandas as pd
import pickle
import math

RESULT_DIR = "result"
RISK_THRESHOLD = 0.75


def calculate_risk(real_x, reco_x, predict_correct, sigma):
    raw_risk = np.linalg.norm(real_x - reco_x) / (2 * sigma * sigma)
    if predict_correct:
        risk = np.exp(-raw_risk)
    else:
        risk = np.exp(raw_risk)
    if risk > 999:
        risk = 999
    return risk


def generate_safe_unsafe_dataset_for_ssl():

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
    tra_dummy_y_unlabeled_y = trs_dummy_y.loc[trs_dummy_y.index.isin(tra_dummy_y_unlabeled.index)]
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

    # sigma values representing L2 norm distance of Xfeatures for each yClass
    sigma_list = []
    for c in tra_dummy_y_labeled.columns:
        sigma = np.linalg.norm(tra_dummy_x_labeled.loc[tra_dummy_x_labeled.index.isin(tra_dummy_y_labeled.loc[tra_dummy_y_labeled[c] == 1].index)].describe().loc['std', :].as_matrix())
        if math.isnan(sigma):
            sigma = np.linalg.norm(tra_dummy_x_labeled.describe().loc['std', :].as_matrix())
        sigma_list.append(sigma)
    # sigma_list = []
    # for c in trs_dummy_y.columns:
    #     sigma = np.linalg.norm(trs_dummy_x.loc[trs_dummy_x.index.isin(trs_dummy_y.loc[trs_dummy_y[c] == 1].index)].describe().loc['std', :].as_matrix())
    #     if math.isnan(sigma):
    #         sigma = np.linalg.norm(trs_dummy_x.describe().loc['std', :].as_matrix())
    #     sigma_list.append(sigma)

    num_ssl_classify_wrong = 0
    num_ssl_classify_correct = 0
    num_ssl_rejected_wrong = 0
    num_ssl_rejected_correct = 0

    ssl_safe_x = pd.DataFrame(columns=tra_dummy_x_unlabeled.columns)
    ssl_safe_y = pd.DataFrame(columns=tra_dummy_y_unlabeled_y.columns)
    not_ssl_safe_x = pd.DataFrame(columns=tra_dummy_x_unlabeled.columns)
    not_ssl_safe_y = pd.DataFrame(columns=tra_dummy_y_unlabeled_y.columns)

    for i in tra_dummy_x_unlabeled.index:

        Xrs = tra_dummy_x_unlabeled.loc[i].as_matrix()
        yrs = tra_dummy_y_unlabeled_y.loc[i].as_matrix()

        prediction = clf.predict(np.reshape(Xrs, (1, len(Xrs))))
        predicted_class = np.argmax(prediction[0])

        # reconstruct back X instance of the predicted y
        reco_x = np.matmul(alpha_k, prediction[0])
        real_x = Xrs

        # predict y base on reconstructed X
        prediction_reco = clf.predict(np.reshape(reco_x, (1, len(Xrs))))
        predicted_class_reco = np.argmax(prediction_reco[0])
        sigma = sigma_list[predicted_class]

        # yXpredicted should be == yXreconstructed
        predict_correct = (predicted_class == predicted_class_reco)
        calc_risk = calculate_risk(real_x, reco_x, predict_correct, sigma)
        real_class = np.argmax(yrs)

        # over risk threshold is considered unsafe for ssl training
        # better to add these as supervised training if possible
        if calc_risk > RISK_THRESHOLD:
            not_ssl_safe_x.loc[i] = tra_dummy_x_unlabeled.loc[i]
            not_ssl_safe_y.loc[i] = tra_dummy_y_unlabeled_y.loc[i]
            if real_class == predicted_class:
                num_ssl_rejected_wrong += 1
            if real_class != predicted_class:
                num_ssl_rejected_correct += 1

        # less than risk threshold is considered safe for ssl training
        if calc_risk < RISK_THRESHOLD:
            ssl_safe_x.loc[i] = tra_dummy_x_unlabeled.loc[i]
            one_hot_encoded = [0] * len(prediction_reco[0])
            one_hot_encoded[predicted_class_reco] = 1
            ssl_safe_y.loc[i] = one_hot_encoded
            if real_class == predicted_class:
                num_ssl_classify_correct += 1
            if real_class != predicted_class:
                num_ssl_classify_wrong += 1

    print("num classified safe but actually unsafe: " + str(num_ssl_classify_wrong))
    print("num classified safe and actually safe: " + str(num_ssl_classify_correct))
    print("num classified unsafe but actually safe: " + str(num_ssl_rejected_wrong))
    print("num classified unsafe and is really unsafe: " + str(num_ssl_rejected_correct))

    merge_safe_y = pd.concat([tra_dummy_y_labeled, ssl_safe_y])
    merge_safe_x = pd.concat([tra_dummy_x_labeled, ssl_safe_x])

    data_path = os.path.join(os.getcwd(), RESULT_DIR, 'dataset-y-ssl-safe.pickle')
    with open(data_path, 'wb') as f:
        pickle.dump(merge_safe_y, f)

    data_path = os.path.join(os.getcwd(), RESULT_DIR, 'dataset-x-ssl-safe.pickle')
    with open(data_path, 'wb') as f:
        pickle.dump(merge_safe_x, f)

    data_path = os.path.join(os.getcwd(), RESULT_DIR, 'dataset-y-ssl-unsafe.pickle')
    with open(data_path, 'wb') as f:
        pickle.dump(not_ssl_safe_y, f)

    data_path = os.path.join(os.getcwd(), RESULT_DIR, 'dataset-x-ssl-unsafe.pickle')
    with open(data_path, 'wb') as f:
        pickle.dump(not_ssl_safe_x, f)

    return merge_safe_y, merge_safe_x, not_ssl_safe_y, not_ssl_safe_x


if __name__ == "__main__":

    # check if labeling of unlabeled training dataset is done
    safe_tra_y_exists = os.path.exists(os.path.join(os.getcwd(), RESULT_DIR, 'dataset-y-ssl-safe.pickle'))
    safe_tra_x_exists = os.path.exists(os.path.join(os.getcwd(), RESULT_DIR, 'dataset-x-ssl-safe.pickle'))
    unsafe_tra_y_exists = os.path.exists(os.path.join(os.getcwd(), RESULT_DIR, 'dataset-y-ssl-unsafe.pickle'))
    unsafe_tra_x_exists = os.path.exists(os.path.join(os.getcwd(), RESULT_DIR, 'dataset-x-ssl-unsafe.pickle'))

    if not (safe_tra_y_exists and safe_tra_x_exists and unsafe_tra_y_exists and unsafe_tra_x_exists):
        merge_safe_y, merge_safe_x, unsafe_y, unsafe_x = generate_safe_unsafe_dataset_for_ssl()
    else:
        with open(os.path.join(os.getcwd(), RESULT_DIR, 'dataset-y-ssl-safe.pickle'), 'rb') as f:
            merge_safe_y = pickle.load(f)
        with open(os.path.join(os.getcwd(), RESULT_DIR, 'dataset-x-ssl-safe.pickle'), 'rb') as f:
            merge_safe_x = pickle.load(f)
        with open(os.path.join(os.getcwd(), RESULT_DIR, 'dataset-y-ssl-unsafe.pickle'), 'rb') as f:
            unsafe_y = pickle.load(f)
        with open(os.path.join(os.getcwd(), RESULT_DIR, 'dataset-x-ssl-unsafe.pickle'), 'rb') as f:
            unsafe_x = pickle.load(f)

        merge_safe_y = pd.DataFrame(merge_safe_y, dtype='int')
        merge_safe_x = pd.DataFrame(merge_safe_x, dtype='int')
        unsafe_y = pd.DataFrame(unsafe_y, dtype='int')
        unsafe_x = pd.DataFrame(unsafe_x, dtype='int')
