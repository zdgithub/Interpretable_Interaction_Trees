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


def conv_tree(temp_tree):
    res = []
    for item in temp_tree:
        if isinstance(item, list):
            res.append(conv_tree(item[0:2]))
        else:
            res.append(item)
    return res
