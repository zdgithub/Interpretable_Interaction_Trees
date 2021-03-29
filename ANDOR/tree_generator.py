import random

random.seed(1)


class TreeGenerator:
    def generate(self, words, type_name):
        if type_name == "random":
            return self.generate_random_tree(words)
        elif type_name == "left-branching":
            return self.generate_left_branching(words)
        elif type_name == "right-branching":
            return self.generate_right_branching(words)
        else:
            raise NotImplementedError()

    @staticmethod
    def generate_random_tree(words):
        res_words = words.copy()
        for i in range(len(words), 2, -1):  # do combination until there is only one node
            # generate the former index of the two words to be combined
            random_index = random.randint(0, i-2)   # generates integer n, 0 <= n <= i-2
            to_be_combined = res_words[random_index: random_index+2]
            res_words[random_index: random_index+2] = [to_be_combined]
        return res_words

    @staticmethod
    def generate_left_branching(words):
        res_words = words.copy()
        for _ in range(len(words), 2, -1):  # do combination until there is only one node
            # generate the former index of the two words to be combined
            chosen_index = 0
            to_be_combined = res_words[chosen_index: chosen_index + 2]
            res_words[chosen_index: chosen_index + 2] = [to_be_combined]
        return res_words

    @staticmethod
    def generate_right_branching(words):
        right_last = False
        res_words = words.copy()
        # the last one is combined at last
        if right_last:
            for _ in range(len(words), 2, -1):  # do combination until there are two node
                # generate the former index of the two words to be combined
                to_be_combined = res_words[-3: -1]
                res_words[-3: -1] = [to_be_combined]
            res_words[-2:] = [res_words[-2:]]
        else:
            for _ in range(len(words), 2, -1):  # do combination until there are two node
                # generate the former index of the two words to be combined
                to_be_combined = res_words[-2:]
                res_words[-2:] = [to_be_combined]
        return res_words
