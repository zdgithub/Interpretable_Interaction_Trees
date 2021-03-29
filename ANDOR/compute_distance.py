import os
import pickle
from calculator import *
from distance_metric import *
from converter import *
from tree_generator import *

pd = PathDistance()
fs = F1Score()
tg = TreeGenerator()

dir_name = "again_sets/ANDOR"
all_pd = []
all_fs = []
all_recall = []
path_list = os.listdir(os.path.join(os.getcwd(), dir_name))
path_list.sort()
for file_name in path_list:
    path = os.path.join(os.getcwd(), dir_name, file_name)
    print(path)
    info = pickle.load(open(path, 'rb'))
    operators = list(info['sentence'])
    groundtruth_tree = gt_tree(operators)

    test_tree = conv_tree(info["tree"])  # for our previous tree structure

    while len(test_tree) == 1:
        test_tree = test_tree[0]
    while len(groundtruth_tree) == 1:
        groundtruth_tree = groundtruth_tree[0]

    print("our tree:", test_tree)
    print("ground truth tree:", groundtruth_tree)
    path_difference = pd.calculate(test_tree, groundtruth_tree)
    recall, f1_score = fs.calculate(test_tree, groundtruth_tree)
    all_pd.append(path_difference)
    all_fs.append(f1_score)
    all_recall.append(recall)

print('-'*50)
print("Average F1 score:", sum(all_fs) / len(all_fs))
print("Average Recall:", sum(all_recall) / len(all_recall))
