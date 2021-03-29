import random
from anytree import Node, RenderTree
from anytree.exporter import DotExporter
from anytree.util import commonancestors
import pickle
random.seed(0)
#

def extract_words(list_words, flattened):
    for item in list_words:
        if isinstance(item, list):
            extract_words(item, flattened)
        else:
            flattened.append(item)
    return flattened


def generate_binary_tree(words):
    res_words = words.copy()
    for i in range(len(words), 2, -1):  # do combination until there is only one node
        # generate the former index of the two words to be combined
        random_index = random.randint(0, i-2)   # generates integer n, 0 <= n <= i-2
        to_be_combined = res_words[random_index: random_index+2]
        res_words[random_index: random_index+2] = [to_be_combined]
    return res_words


def build_tree(temp_word, nonleaf_list, leaf_list, p):
    """
    build a tree
    hierarchical list of list as input
    :param temp_word: the str/list of list that is processed now
    :param nonleaf_list: list to store nonleaf nodes, initially []
    :param leaf_list:  list to store leaf nodes, initially []
    :param p: parent of the temp node
    :return: list of nonleaf nodes and list of leaf nodes
    """
    if isinstance(temp_word, list):  # nonleaf node
        nonleaf_counter = len(nonleaf_list)
        temp_node = Node("*" + str(nonleaf_counter), parent=p)  # nonleaf node is named by *+number
        nonleaf_list.append(temp_node)
        for child in temp_word:
            build_tree(child, nonleaf_list, leaf_list, nonleaf_list[nonleaf_counter])
    else:  # leaf node
        leaf_counter = len(leaf_list)
        temp_node = Node(str(leaf_counter) + ":" + temp_word, parent=p)
        leaf_list.append(temp_node)
    return nonleaf_list, leaf_list


def path_length_to_ancestor(node, ancestor):
    temp_node = node
    path_length = 0
    while temp_node != ancestor:
        temp_node = temp_node.parent
        path_length += 1
    return path_length


def list_to_leaves(list_tree, tree_name, display=True):
    """

    :param list_tree: list of list that indicates hierarchical relationship
    :param tree_name: str
    :param display: bool
    :return: list of leaf nodes in the same order as that in the original sentence
    """
    root = Node(tree_name)
    non_leaves, leaves = build_tree(list_tree, nonleaf_list=[], leaf_list=[], p=root)
    if display:
        for pre, fill, node in RenderTree(root):
            print("{0}{1}".format(pre, node.name))
        # DotExporter(root).to_picture("./res/{0}.png".format(tree_name))
    return leaves


def build_tree_model(temp_word, nonleaf_list, leaf_list, p):
    if isinstance(temp_word, list):  # nonleaf node, composed of (left, right, coalition_information)
        nonleaf_counter = len(nonleaf_list)
        left, right, information = temp_word
        coalition_label = ""
        for (key, value) in information.items():
            if key=='phi_a' or key=='phi_b':
                continue
            if isinstance(value, list):
                coalition_label += "{0}: {1}\n".format('Bl', round(value[0], 4))
                coalition_label += "{0}: {1}\n".format('Br', round(value[1], 4))
                continue
            coalition_label += "{0}: {1}\n".format(key, round(value, 4))
        temp_node = Node(coalition_label, parent=p)  
        nonleaf_list.append(temp_node)
        build_tree_model(left, nonleaf_list, leaf_list, nonleaf_list[nonleaf_counter])
        build_tree_model(right, nonleaf_list, leaf_list, nonleaf_list[nonleaf_counter])
    elif isinstance(temp_word, str):  # leaf node
        leaf_counter = len(leaf_list)
        temp_node = Node(str(leaf_counter) + ":" + temp_word, parent=p)
        leaf_list.append(temp_node)
    else:  # discard coalition information
        pass
    return nonleaf_list, leaf_list



def list_to_leaves_model(list_tree, tree_name, display=True):
    root = Node(tree_name)
    non_leaves, leaves = build_tree_model(list_tree, nonleaf_list=[], leaf_list=[], p=root)
    if display:
        # for pre, fill, node in RenderTree(root):
            # print("{0}{1}".format(pre, node.name))
        print("tree name:", tree_name)
        print(root)
        DotExporter(root).to_picture(tree_name)
    return non_leaves, leaves

def leaves_to_vector(leaves):
    """
    given a list of leaf nodes, return the pairwise path length in the order
    (0,1), (0,2) ... ,(0, i), (1, 2), (1, 3) ,..., (i-1, i)
    where i is the index of the last leaf node
    :param leaves:list(nodes)
    :return:list(int)
    """
    vector = []
    num_of_leaves = len(leaves)
    for i in range(num_of_leaves):
        for j in range(i+1, num_of_leaves):
            node1 = leaves[i]
            node2 = leaves[j]
            ancestor_list = list(commonancestors(node1, node2))
            # The node with maximum string length corresponds to the lowest common ancestor(LCA)
            # Because the string is composed of path from root to the node
            # For example, in ["Node('/tree1')", "Node('/tree1/*0')", "Node('/tree1/*0/*1')"],
            # "Node('/tree1/*0/*1')" has maximum length, and must be LCA
            len_str_list = [len(str(item)) for item in ancestor_list]
            lowest_common_ancestor = ancestor_list[len_str_list.index(max(len_str_list))]
            path_length1 = path_length_to_ancestor(node1, lowest_common_ancestor)  # path length from node1 to LCA
            path_length2 = path_length_to_ancestor(node2, lowest_common_ancestor)  # path length from node2 to LCA
            total_length = path_length1 + path_length2
            vector.append(total_length)
            # print("{0}\t{1}\t{2}".format
            #       (leaves[i].name, leaves[j].name, lowest_common_ancestor.name))
            # print("total path length: ", total_length)
    return vector


def vector_distance(vector1, vector2, p=2):
    """
    calculate the Euclidean distance between the two vectors
    :param vector1: list(int)
    :param vector2: list(int)
    :param p: int, indicates l_p norm
    :return: float
    """
    assert len(vector1) == len(vector2)
    total = 0
    for i in range(len(vector1)):
        temp = (vector1[i] - vector2[i]) ** p
        total += temp
    res = total ** (1/p)
    return res


def substitude(sentence_list, temp_tree):
    res = []
    for item in temp_tree:
        if isinstance(item, int):
            res.append(sentence_list[item])
        elif isinstance(item, list):
            res.append(substitude(sentence_list, item))
        else:
            res.append(item)
    return res
            
"""
task = "sst-2"
for i in range(20):
    save_path = "0129_v3/{0}/stn_{1}.pkl".format(task, str(i))
    with open(save_path, "rb") as f:
        contents = pickle.load(f)
        # print(contents.keys())
        sentence = contents["sentence"]
        tree = contents["tree"][0]
        print("sentence:", sentence)
        sentence_tree = substitude(sentence, tree)
        # print(sentence_tree)
        list_to_leaves_model(sentence_tree, "{0}_{1}.png".format(task, str(i)))
        # nonleaves[0].name = "hello"
"""

