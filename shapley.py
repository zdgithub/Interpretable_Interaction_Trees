import numpy as np
import time
from itertools import combinations, chain
import scipy.special
from keras.utils import to_categorical
import math

np.random.seed(0)


def turn_list(s):
    if type(s) == list:
        return s
    elif type(s) == int:
        return [s]


def powerset(S, order=None):
    """
    Compute the power set(a set of all subsets) of a set.
    :param S: a set represented by list
    :param order: max number of selected elements from S
    :return: a dictionary that partitions subsets by cardinality.
    keys are the cardinality and values are lists of subsets.
    """
    if order is None:
        order = len(S)

    return {r:list(combinations(S, r)) for r in range(order + 1)}


def construct_dict_sampleshapley(slist, cIdx, a_len):
    """Construct the position dict of sample shapley"""
    d = len(slist)
    cur = turn_list(slist[cIdx])  # current coalition
    sd = len(cur)

    positions_dict = {(i, fill):[] for i in range(sd+1) for fill in [0, 1]}
    # sample m times
    m = 1000
    for cnt in range(m):
        perm = np.random.permutation(d)
        preO = []
        for idx in perm:
            if idx != cIdx:
                preO.append(turn_list(slist[idx]))
            else:
                break

        preO_list = list(chain.from_iterable(preO))
        pos_excluded = np.sum(to_categorical(preO_list, num_classes=a_len), axis=0)
        pos_included = pos_excluded + np.sum(to_categorical(turn_list(slist[cIdx]), num_classes=a_len), axis=0)
        positions_dict[(0, 0)].append(pos_excluded)
        positions_dict[(0, 1)].append(pos_included)
        for j in range(sd):
            subperm = np.random.permutation(d)
            subpreO = []
            for sidx in subperm:
                if sidx != cIdx:
                    subpreO.append(turn_list(slist[sidx]))
                else:
                    break

            tmp = cur[j]  # the elems in set S
            subpreO_list = list(chain.from_iterable(subpreO))
            pos_exc = np.sum(to_categorical(subpreO_list, num_classes=a_len), axis=0)
            pos_inc = pos_exc + np.sum(to_categorical(turn_list(tmp), num_classes=a_len), axis=0)
            positions_dict[(j + 1, 0)].append(pos_exc)
            positions_dict[(j + 1, 1)].append(pos_inc)

    keys, values = positions_dict.keys(), positions_dict.values()
    values = [np.array(value) for value in values]
    positions = np.concatenate(values, axis=0)

    key_to_idx = {}
    count = 0
    for i, key in enumerate(keys):
        key_to_idx[key] = list(range(count, count + len(values[i])))
        count += len(values[i])

    return positions_dict, key_to_idx, positions


def singleton_shapley(slist):
    d = len(slist)
    a_len = d
    m = 1000
    null_set = 1 - np.sum(to_categorical(list(range(d)), num_classes=a_len), axis=0)

    positions_dict = {(i, fill):[] for i in range(d) for fill in [0, 1]}

    for cnt in range(m):
        perm = np.random.permutation(d)
        pos_exc = null_set

        preO = []
        for idx in perm:
            preO.append([idx])
            preO_list = list(chain.from_iterable(preO))
            pos_inc = np.sum(to_categorical(preO_list, num_classes=a_len), axis=0)
            positions_dict[(idx, 0)].append(pos_exc)
            positions_dict[(idx, 1)].append(pos_inc)
            pos_exc = pos_inc

    keys, values = positions_dict.keys(), positions_dict.values()
    values = [np.array(value) for value in values]
    positions = np.concatenate(values, axis=0)

    key_to_idx = {}
    count = 0
    for i, key in enumerate(keys):
        key_to_idx[key] = list(range(count, count + len(values[i])))
        count += len(values[i])

    return key_to_idx, positions


def explain_shapley(predict, d, x, batch_dict, key_to_idx):
    """
    Compute the importance score/shapley value
    :param predict: network function
    :param d: d points needed to compute shapley value in current coalition
    :param x: feature of input x
    """
    f_logits = predict(batch_dict)
    logits = predict(x)
    discrete_probs = np.eye(len(logits[0]))[np.argmax(logits, axis=-1)]
    vals = np.sum(discrete_probs * f_logits, axis=1)

    # key_to_idx[key]: list of indices in original position.
    key_to_val = {key: np.array([vals[idx] for idx in key_to_idx[key]]) for key in key_to_idx}

    # Compute importance scores.
    phis = np.zeros(d)
    for i in range(d):
        phis[i] = np.mean(key_to_val[(i, 1)] - key_to_val[(i, 0)])

    return phis


def compute_scores(slist, cIdx, feature, a_len, predict, method=None):
    """
    Compute shapley values of each feature(including coalition) of slist, e.g. [0,1,[2,3],4]
    :param slist: now pair combination list of x
    :param cIdx: now pair combination start position e.g. 2
    :param feature: input features of x
    :param a_len: # of words in primitive x without padding and [CLS],[SEP]
    :return:
    """
    flag_raw = False
    if method == 'SampleShapley':
        positions_dict, key_to_idx, positions = construct_dict_sampleshapley(slist, cIdx, a_len)
    else:
        flag_raw = True
        # only compute primitive shapley
        key_to_idx, positions = singleton_shapley(slist)

    #  tokens: [CLS] ... [SEP]
    real_ids = feature.input_ids[1:a_len+1]
    inputs = np.array(real_ids) * positions

    batch_in = {'input_ids':[], 'input_mask':[], 'segment_ids':[], 'label_ids':[]}
    for j in range(inputs.shape[0]):
        input_ids = [feature.input_ids[0]]
        input_ids = input_ids + list(inputs[j])
        input_ids = input_ids + feature.input_ids[(a_len+1):]

        batch_in['input_ids'].append(input_ids)
        batch_in['input_mask'].append(feature.input_mask)
        batch_in['segment_ids'].append(feature.segment_ids)
        batch_in['label_ids'].append(feature.label_id)

    batch_dict = {
        'input_ids': np.array(batch_in['input_ids']),
        'input_mask': np.array(batch_in['input_mask']),
        'segment_ids': np.array(batch_in['segment_ids']),
        'label_ids': np.array(batch_in['label_ids']),
    }
    x = {
        'input_ids': np.array([feature.input_ids]),
        'input_mask': np.array([feature.input_mask]),
        'segment_ids': np.array([feature.segment_ids]),
        'label_ids': np.array([feature.label_id])
    }

    if flag_raw:
        d = a_len
    else:
        d = len(turn_list(slist[cIdx])) + 1

    shapvals = explain_shapley(predict, d, x, batch_dict, key_to_idx)

    return shapvals




