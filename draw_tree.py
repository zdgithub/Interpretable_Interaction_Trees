import pickle
import numpy as np
from load_tree import *
import os


root = 'binary_trees'
for task in ['SST-2']:
    stns = os.listdir(os.path.join(root, task))
    for index in range(len(stns)):
        path_name = os.path.join(root, task, "stn_{}.pkl".format(index))
        contents = pickle.load(open(path_name, "rb"))
        tree = contents["tree"][0]
        sentence = contents["sentence"]
        print("sentence:", sentence)
        sentence_tree = substitude(sentence, tree)
        # print(sentence_tree)
        tree_dir = os.path.join(root, "draw_trees", task)
        if not os.path.exists(tree_dir):
            os.makedirs(tree_dir)
        tree_name = os.path.join(tree_dir, "{}.png".format(str(index)))
        tree_name = str(tree_name).replace("\\", "/")
        print(tree_name)

        list_to_leaves_model(sentence_tree, tree_name)

