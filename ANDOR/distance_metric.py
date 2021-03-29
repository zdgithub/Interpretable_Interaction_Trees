from anytree import Node
from anytree.util import commonancestors
import nltk


class DistanceMetric:
    def __init__(self, metric_name):
        self.metric_name = metric_name

    def calculate(self, sentence_tree1, sentence_tree2):
        raise NotImplementedError()


class PathDistance(DistanceMetric):
    def __init__(self):
        DistanceMetric.__init__(self, "path_distance")

    @classmethod
    def _list_to_leaves(cls, list_tree):
            """
            :param list_tree: list of list that indicates hierarchical relationship
            :return: list of leaf nodes in the same order as that in the original sentence
            """
            root = Node("tree")
            non_leaves, leaves = cls._build_tree(list_tree, nonleaf_list=[], leaf_list=[], p=root)
            return leaves

    @classmethod
    def _build_tree(cls, temp_word, nonleaf_list, leaf_list, p):
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
            # for child in temp_word:
            # build_tree(child, nonleaf_list, leaf_list, nonleaf_list[nonleaf_counter])
            for child in temp_word:
                cls._build_tree(child, nonleaf_list, leaf_list, nonleaf_list[nonleaf_counter])
        else:  # leaf node
            leaf_counter = len(leaf_list)
            temp_node = Node(str(leaf_counter) + ":" + str(temp_word), parent=p)
            leaf_list.append(temp_node)
        return nonleaf_list, leaf_list

    @classmethod
    def _leaves_to_vector(cls, leaves):
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
            for j in range(i + 1, num_of_leaves):
                node1 = leaves[i]
                node2 = leaves[j]
                ancestor_list = list(commonancestors(node1, node2))
                # The node with maximum string length corresponds to the lowest common ancestor(LCA)
                # Because the string is composed of path from root to the node
                # For example, in ["Node('/tree1')", "Node('/tree1/*0')", "Node('/tree1/*0/*1')"],
                # "Node('/tree1/*0/*1')" has maximum length, and must be LCA
                len_str_list = [len(str(item)) for item in ancestor_list]
                lowest_common_ancestor = ancestor_list[len_str_list.index(max(len_str_list))]
                # path length from node to LCA
                path_length1 = cls._path_length_to_ancestor(node1, lowest_common_ancestor)
                path_length2 = cls._path_length_to_ancestor(node2, lowest_common_ancestor)
                total_length = path_length1 + path_length2
                vector.append(total_length)
                # print("{0}\t{1}\t{2}".format
                #       (leaves[i].name, leaves[j].name, lowest_common_ancestor.name))
                # print("total path length: ", total_length)
        return vector

    @staticmethod
    def _vector_distance(vector1, vector2, p=2):
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
        res = total ** (1 / p)
        return res

    @staticmethod
    def _path_length_to_ancestor(node, ancestor):
        temp_node = node
        path_length = 0
        while temp_node != ancestor:
            temp_node = temp_node.parent
            path_length += 1
        return path_length

    def calculate(self, sentence_tree1, sentence_tree2):
        leaves_model1 = self._list_to_leaves(sentence_tree1)
        leaves_model2 = self._list_to_leaves(sentence_tree2)
        vector_model1 = self._leaves_to_vector(leaves_model1)
        vector_model2 = self._leaves_to_vector(leaves_model2)
        return self._vector_distance(vector_model1, vector_model2)


class F1Score(DistanceMetric):
    def __init__(self):
        DistanceMetric.__init__(self, "f1_score")

    @classmethod
    def _get_brackets(cls, tree, idx=0):
        brackets = set()
        if isinstance(tree, list) or isinstance(tree, nltk.Tree):
            for node in tree:
                node_brac, next_idx = cls._get_brackets(node, idx)
                if next_idx - idx > 1:
                    brackets.add((idx, next_idx))
                    brackets.update(node_brac)
                idx = next_idx
            return brackets, idx
        else:
            return brackets, idx + 1

    def calculate(self, sentence_tree1, sentence_tree2):
        model1_out, _ = self._get_brackets(sentence_tree1)
        model2_out, _ = self._get_brackets(sentence_tree2)
        print("our tree:", model1_out)
        print("ground truth:", model2_out)
        overlap = model1_out.intersection(model2_out)
        precision = float(len(overlap)) / (len(model1_out) + 1e-8)
        recall = float(len(overlap)) / (len(model2_out) + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        print("precision={}, recall={}, f1={}".format(precision, recall, f1))
        return recall, f1

