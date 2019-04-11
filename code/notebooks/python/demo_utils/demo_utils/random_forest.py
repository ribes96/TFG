#!/usr/bin/python3

# Este documento hace las ejecuciones de Random Forest con los
# datasets

from sklearn.ensemble import RandomForestClassifier
import numpy as np
from mnist import MNIST
from sklearn.preprocessing import StandardScaler
from demo_utils.general import get_data
import time
import json

from demo_utils.general import SUPPORTED_DATASETS
RF_SUPPORTED_DATASETS = SUPPORTED_DATASETS + ['fashion_mnist']


def get_fashion_data():
    data_dir = '/home/hobber/fashion_mnist'
    mndata = MNIST(data_dir)
    data_train, target_train = mndata.load_training()
    data_test, target_test = mndata.load_testing()

    scaler = StandardScaler()
    scaler.fit(data_train)
    data_train = scaler.transform(data_train)
    data_test = scaler.transform(data_test)

    d = {
        'data_train': np.array(data_train),
        'data_test': np.array(data_test),
        'target_train': np.array(target_train),
        'target_test': np.array(target_test),
    }
    return d


def get_mnist_data():
    data_dir = '/home/hobber/mnist'
    mndata = MNIST(data_dir)
    data_train, target_train = mndata.load_training()
    data_test, target_test = mndata.load_testing()

    scaler = StandardScaler()
    scaler.fit(data_train)
    data_train = scaler.transform(data_train)
    data_test = scaler.transform(data_test)

    d = {
        'data_train': np.array(data_train),
        'data_test': np.array(data_test),
        'target_train': np.array(target_train),
        'target_test': np.array(target_test),
    }
    return d

    #####################
    # segment
    # covertype
    # digits
    # fall_detection
    # pen_digits
    # satellite
    # vowel
    #
    # mnist
    # fashion_mnist
    #######################


def fun(dts_name):
    o_filename = f'experimental_results/random_forest/{dts_name}.json'
    if dts_name == 'mnist':
        data = get_mnist_data()
    elif dts_name == 'fashion_mnist':
        data = get_fashion_data()
    else:
        data = get_data(dataset_name=dts_name, prop_train=2/3, n_ins=5000)

    rf = RandomForestClassifier(n_estimators=50)

    data_train = data['data_train']
    data_test = data['data_test']
    target_train = data['target_train']
    target_test = data['target_test']

    print('Empieza el experimento')
    time0 = time.perf_counter()
    rf.fit(data_train, target_train)
    time1 = time.perf_counter()
    c_time = time1 - time0
    print('Termina el experimento')

    train_score = rf.score(data_train, target_train)
    test_score = rf.score(data_test, target_test)

    info = {
        "box_name": "none",
        "description": f"A normal RF ({dts_name})",
        "gamma": None,
        "label": "RF ",
        "model_name": "rf",
        "model_param": {},
        "test_score": test_score,
        "time": c_time,
        "train_score": train_score,
        }
    print(info)
    with open(o_filename, 'w') as f:
        json.dump([info], f, indent=4, sort_keys=True)


def run_all_dts():
    fun('segment')
    fun('covertype')
    fun('digits')
    fun('fall_detection')
    fun('pen_digits')
    fun('satellite')
    fun('vowel')
    fun('mnist')
    fun('fashion_mnist')
