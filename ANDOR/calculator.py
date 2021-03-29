import random
import numpy as np
from copy import deepcopy

random.seed(0)


class Calculator:
    def __init__(self, dictionary):
        self.d = dictionary  # higher number means higher priority
        self.operands = None  # real value for each parameter
        self.operators = None  # operators encoded by dictionary

    def update_dictionary(self, new_dictionary):
        self.d = new_dictionary

    def update_operands(self, new_operands):
        self.operands = deepcopy(new_operands)

    def update_operators(self, new_operators):
        self.operators = deepcopy(new_operators)

    def show_expression(self):
        assert len(self.operands) == len(self.operators) + 1
        if len(self.operators) == 0:
            print(self.operands[0])
        else:
            expression_list = []
            for i in range(len(self.operators)):
                expression_list.append(self.operands[i])
                expression_list.append(self.d[self.operators[i]])
            expression_list.append(self.operands[-1])
            print(" ".join(str(term) for term in expression_list))

    def calculate(self):
        # self.show_expression()
        while len(self.operators) > 0:
            chosen_operator = max(self.operators)
            index_chosen_operator = self.operators.index(chosen_operator)
            index_operand1 = index_chosen_operator
            index_operand2 = index_chosen_operator + 1
            operand1, operand2 = self.operands[index_operand1], self.operands[index_operand2]
            if self.d[chosen_operator] == "+":
                temp_res = operand1 + operand2
            elif self.d[chosen_operator] == "*":
                temp_res = operand1 * operand2
            elif self.d[chosen_operator] == "AND":
                temp_res = operand1 * operand2
            elif self.d[chosen_operator] == "OR":
                temp_res = min(operand1 + operand2, 1)
            else:
                print("invalid input")
                exit(1)
            self.operators.pop(index_chosen_operator)
            self.operands.pop(index_operand2)
            self.operands.pop(index_operand1)
            self.operands.insert(index_operand1, temp_res)
            # self.show_expression()
        return self.operands[0]


# generate ground-truth n-ary tree
def gt_tree(ops):
    vals = list(range(len(ops) + 1))
    previous_index = 0  # the index of the last operator that has been examined
    offset = 0  # offset for index due to combination
    while previous_index < len(ops) - 1:
        first_index = last_index = -1
        start_index = previous_index
        for temp_index in range(start_index, len(ops)):
            if first_index == -1 and ops[temp_index] == 1:
                first_index = last_index = temp_index
            elif ops[temp_index] == 1:
                last_index = temp_index
            else:
                if first_index != -1:
                    to_be_combined = vals[first_index-offset:last_index-offset+2]
                    vals[first_index-offset: last_index-offset+2] = [to_be_combined]
                    offset += (last_index - first_index + 1)
                    previous_index = temp_index
                    break
            previous_index = temp_index
        if last_index == len(ops) - 1:
            to_be_combined = vals[first_index - offset:last_index - offset + 2]
            vals[first_index - offset: last_index - offset + 2] = [to_be_combined]
            offset += (last_index - first_index + 1)

    return vals


