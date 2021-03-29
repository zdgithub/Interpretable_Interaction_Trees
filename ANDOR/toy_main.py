import numpy as np
from itertools import combinations, chain
from keras.utils import to_categorical
import time
import os
import math
from calculator import Calculator
import pickle
import random


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

    return {r: list(combinations(S, r)) for r in range(order + 1)}


def get_allsets(a_len):
    """
    :param a_len: length
    :return: list of subsets arrays
    """
    a = powerset(np.arange(a_len))
    b = list(chain.from_iterable(a.values()))
    # print(b)
    all_sets = []
    all_sets.append(np.zeros(a_len))
    for i, item in enumerate(b):
        if i == 0:
            continue
        indexes = np.array(item)
        t = np.zeros(a_len)
        t[indexes] = 1
        all_sets.append(t)

    return all_sets

def get_equations(n):
    """
    :param n: length of operators
    :return: all equations
    """
    return get_allsets(n)


def compute_accurate_score(slist, cIdx, a_len, sample, calc, operators):
    d = len(slist)
    cur = turn_list(slist[cIdx])
    sd = len(cur)
    other_len = d - 1

    other_list = []
    for i in range(d):
        if i != cIdx:
            other_list.append(i)

    positions_dict = {(i, fill): [] for i in range(sd + 1) for fill in [0, 1]}
    weight_list = []

    for num in range(other_len + 1):
        # num is the size of subset
        subset = list(combinations(other_list, num))
        for item in subset:
            exc = np.zeros_like(list(range(a_len)))
            inc = np.zeros_like(list(range(a_len)))
            perm = []
            for i in item:
                perm += turn_list(slist[i])
            if num > 0:
                exc[perm] = 1
                inc[perm] = 1
            for j in cur:
                inc[j] = 1
            weight = 1.0 * math.factorial(num) * math.factorial(d - 1 - num) / math.factorial(d)
            weight_list.append(weight)
            positions_dict[(0, 0)].append(exc)
            positions_dict[(0, 1)].append(inc)

            for j in range(sd):
                tmp = cur[j]
                pos_exc = exc
                inc_i = np.zeros_like(list(range(a_len)))
                inc_i[tmp] = 1
                pos_inc = exc + inc_i
                positions_dict[(j+1, 0)].append(pos_exc)
                positions_dict[(j+1, 1)].append(pos_inc)

    weight_list = np.array(weight_list)

    keys, values = positions_dict.keys(), positions_dict.values()
    values = [np.array(value) for value in values]
    positions = np.concatenate(values, axis=0)

    key_to_idx = {}
    count = 0
    for i, key in enumerate(keys):
        key_to_idx[key] = list(range(count, count + len(values[i])))
        count += len(values[i])

    # compute the function result of the sample
    vals = []
    features = sample * positions
    for feature in features:
        calc.update_operators(operators.tolist())
        calc.update_operands(feature.tolist())
        out = calc.calculate()
        vals.append(out)

    key_to_val = {key: np.array([vals[idx] for idx in key_to_idx[key]]) for key in key_to_idx}

    # Compute importance scores.
    phis = np.zeros(sd + 1)
    for i in range(sd + 1):
        margin = (key_to_val[(i, 1)] - key_to_val[(i, 0)])
        phis[i] = np.sum(margin * weight_list)

    return phis


def get_sentences(a_len, operators, sample, calc, save_path):
    st = time.time()
    pre_slist = list(range(a_len))
    output_tree = list(range(a_len))
    tree_values = []

    for h in range(a_len - 1):
        pre_slen = len(pre_slist)
        totcombs = []
        ratios = []
        stn = {}

        # compute B, phi{a,b,...} for each point
        tot_values = {}
        for k in range(pre_slen):
            scores = compute_accurate_score(pre_slist, k, a_len, sample, calc, operators)
            if len(scores) == 2:
                b = 0
                subtree = [b, scores[0:1]]
            else:
                b = scores[0] - np.sum(scores[1:])
                subtree = [b, scores[1:]]
            tot_values[k] = subtree

        locs = []
        # there are n-1 pair combinations
        for j in range(pre_slen - 1):
            coal = turn_list(pre_slist[j]) + turn_list(pre_slist[j + 1])
            now_slist = pre_slist[:j]  # elems before j
            now_slist.append(coal)
            if j + 2 < pre_slen:
                now_slist = now_slist + pre_slist[j + 2:]  # elems after j+1

            totcombs.append(now_slist)
            # compute shapley values of now pair combination
            score = compute_accurate_score(now_slist, j, a_len, sample, calc, operators)

            nowb = score[0] - np.sum(score[1:])
            nowphis = score[1:]

            lt = tot_values[j][1]
            rt = tot_values[j + 1][1]
            avgphis = (nowphis + np.concatenate((lt, rt))) / 2
            len_lt = lt.shape[0]

            b_lt = tot_values[j][0]
            b_rt = tot_values[j + 1][0]
            b_local = nowb - b_lt - b_rt
            if abs(b_local) < 1e-10:
                b_local = 0.0
            contri_lt = b_lt + np.sum(avgphis[:len_lt])
            contri_rt = b_rt + np.sum(avgphis[len_lt:])

            locs.append([b_local, contri_lt, contri_rt, b_lt, b_rt, nowb])

        for j in range(pre_slen - 1):
            loss = 0.0
            if j - 1 >= 0:
                loss = loss + abs(locs[j - 1][0])

            if j + 2 < pre_slen:
                loss = loss + abs(locs[j + 1][0])

            all_info = loss + abs(locs[j][0]) + abs(locs[j][1]) + abs(locs[j][2])
            if all_info == 0:
                metric = 0
                sub_metric = 0
            else:
                metric = abs(locs[j][0]) / all_info
                sub_metric = loss / all_info

            ratios.append(metric)

            stn[j] = {'r': metric,
                      's': sub_metric,
                      'Bbetween': locs[j][0],
                      'Bl': locs[j][3],
                      'Br': locs[j][4],
                      'B([S])': locs[j][5],
                      }
        stn['base_B'] = tot_values
        coalition = np.argmax(np.array(ratios))
        pre_slist = totcombs[coalition]
        stn['maxIdx'] = coalition
        stn['after_slist'] = pre_slist
        print('after_slist:', pre_slist)

        # generate a new nested list by adding elements into a empty list---------
        tmp_list = []
        for z in range(len(output_tree)):
            if z == coalition:
                tmp_list.append(list((output_tree[z], output_tree[z + 1], stn[z])))
            elif z == coalition + 1:
                continue
            else:
                tmp_list.append(output_tree[z])
        output_tree = tmp_list.copy()

        tree_values.append(stn)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    filename = "stn_{}.pkl".format(''.join(str(e) for e in operators))
    save_pkl = os.path.join(save_path,  filename)
    with open(save_pkl, "wb") as f:
        contents = {"sentence": operators,
                    "tree": output_tree
                    }
        pickle.dump(contents, f)

    print("Time spent is {}".format(time.time() - st))
    print("---------------------------------------------------")


if __name__ == "__main__":
    random.seed(0)
    a_len = 11
    equations = get_equations(a_len-1)  # the number of models is 2^N
    d = {1: "AND", 0: "OR"}  # AND first; or {1: "OR", 0: "AND"} for OR first
    calc = Calculator(d)
    save_dir = "again_sets/ANDOR"
    for eq in equations:
        operators = eq.astype(int)
        print('Equation:', [d[op] for op in operators])
        sample = np.ones(a_len)
        get_sentences(a_len, operators, sample, calc, save_dir)




















