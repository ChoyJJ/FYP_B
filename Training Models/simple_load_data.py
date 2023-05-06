'''
Code and dataset from https://www.kaggle.com/datasets/llkihn/ddsm-cbis-patch?select=simple_load_data.py accessed on 25 February 2023
'''

import os
import random
# from typing import List
from collections import defaultdict, Counter

import pandas as pd
import numpy as np
# import lightning as pl
# import matplotlib.pyplot as plt

# from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

DATASET_PATH = '/mnt/d/CBIS_DDSM_Patch'


def label_as_text(label: int):
    if label == 0:
        return 'BENIGN MASS'
    elif label == 1:
        return 'BENIGN CALCIFICATION'
    elif label == 2:
        return 'MALIGNANT MASS'
    elif label == 3:
        return 'MALIGNANT CALCIFICATION'


def load_all_files(data_directory: str = DATASET_PATH):
    """Load the data in the simplest way"""

    train_metadata: pd.DataFrame = pd.read_hdf(os.path.join(data_directory, 'train_meta.h5'), key='data')
    test_metadata: pd.DataFrame = pd.read_hdf(os.path.join(data_directory, 'test_meta.h5'), key='data')

    train_data = np.array(np.memmap(os.path.join(data_directory, 'train_data.npy'), dtype='uint8', mode='r',
                                    shape=(len(train_metadata), 900, 900)))
    test_data = np.array(np.memmap(os.path.join(data_directory, 'test_data.npy'), dtype='uint8', mode='r',
                                   shape=(len(test_metadata), 900, 900)))

    train_labels = np.array(np.memmap(os.path.join(data_directory, 'train_labels.npy'), dtype='uint8', mode='r',
                                      shape=(len(train_metadata),)))
    test_labels = np.array(np.memmap(os.path.join(data_directory, 'test_labels.npy'), dtype='uint8', mode='r',
                                     shape=(len(test_metadata),)))

    return (train_data, train_labels, train_metadata), (test_data, test_labels, test_metadata)


def stratified_group_k_fold(X, y, groups, k, seed=None) -> (np.ndarray, np.ndarray):
    """
    Borrowed from https://www.kaggle.com/jakubwasikowski/stratified-group-k-fold-cross-validation
    Combination of stratified and group k fold
    :param X:
    :param y:
    :param groups:
    :param k:
    :param seed:
    :return:
    """
    labels_num = np.max(y) + 1
    y_counts_per_group = defaultdict(lambda: np.zeros(labels_num))
    y_distr = Counter()
    for label, g in zip(y, groups):
        y_counts_per_group[g][label] += 1
        y_distr[label] += 1

    y_counts_per_fold = defaultdict(lambda: np.zeros(labels_num))
    groups_per_fold = defaultdict(set)

    def eval_y_counts_per_fold(y_counts, fold):
        y_counts_per_fold[fold] += y_counts
        std_per_label = []
        for label in range(labels_num):
            label_std = np.std([y_counts_per_fold[i][label] / y_distr[label] for i in range(k)])
            std_per_label.append(label_std)
        y_counts_per_fold[fold] -= y_counts
        return np.mean(std_per_label)

    groups_and_y_counts = list(y_counts_per_group.items())
    random.Random(seed).shuffle(groups_and_y_counts)

    for g, y_counts in sorted(groups_and_y_counts, key=lambda x: -np.std(x[1])):
        best_fold = None
        min_eval = None
        for i in range(k):
            fold_eval = eval_y_counts_per_fold(y_counts, i)
            if min_eval is None or fold_eval < min_eval:
                min_eval = fold_eval
                best_fold = i
        y_counts_per_fold[best_fold] += y_counts
        groups_per_fold[best_fold].add(g)

    all_groups = set(groups)
    for i in range(k):
        train_groups = all_groups - groups_per_fold[i]
        test_groups = groups_per_fold[i]

        train_indices = [i for i, g in enumerate(groups) if g in train_groups]
        test_indices = [i for i, g in enumerate(groups) if g in test_groups]

        yield np.array(train_indices), np.array(test_indices)


def divide_into_k_folds(metadata: pd.DataFrame, k: int, random_seed: int = 42):
    """
    Use for k fold cross validation, divides data in such way, that patient only appears in one fold
    :param metadata:
    :param k: number of folds
    :param random_seed: seed to fix randomness of selection
    :return: list of folds, each fold contains list of ids
    """
    ids = np.arange(len(metadata))
    labels = metadata['new_labels'].values
    groups = np.array(metadata['patient_id'].values)
    folds = list(
        np.array(val_ids) for _, val_ids in stratified_group_k_fold(ids, labels, groups, k=k, seed=random_seed))

    return folds
