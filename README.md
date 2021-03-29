## Building Interpretable Interaction Trees for Deep NLP Models
Official implementation for the paper [Building Interpretable Interaction Trees for Deep NLP Models](https://arxiv.org/abs/2007.04298) (AAAI 2021).

## Requirements
- python 3.6
- nltk
- tensorflow-gpu==1.14.0
- keras==2.3.1
- anytree==2.8.0
- graphviz==0.8.4

## Models
Download the pre-trained models on SST-2 and CoLA [here](https://drive.google.com/drive/folders/1t0TNRLy2RlN7igqZxFIY5ZbmPqbZ-sDl?usp=sharing).
- SST-2: 
    - eval_accuracy: 92.32%
    - checkpoint: `models/sst-2/model.ckpt-6313`

- CoLA: 
    - eval_accuracy: 82.26%
    - checkpoint: `models/cola/model.ckpt-801`

## Build interaction trees on NLP tasks
1. To build interaction trees on some examples, you can run
```
python main.py --task_name sst-2
```
then results of the interaction tree for each sentence will be saved in folder `binary_trees`.

2. To visualize the extracted interaction tree structure, you should run
```
python draw_tree.py
```
and the pictures of tree structures will be saved in folder `draw_trees`.


## Toy Task - ANDOR
To evaluate the correctness of the extracted interactions, we construct a dataset with
 ground-truth interactions between the inputs. Each example only contains AND operations and OR operations.

1. To compute interactions among input variables in the ANDOR dataset you can run
```
cd ANDOR
python toy_main.py
```

2. Then you can compute the **F1-socre** and **Recall** between the extracted interactions and their ground-truths:
```
python compute_distance.py
```


## Citation
If you find our work useful for your research, please cite the following paper:
```
@inproceedings{zhang2021building,
  title={Building interpretable interaction trees for deep nlp models},
  author={Zhang, Die and Zhang, Hao and Zhou, Huilin and Bao, Xiaoyi and Huo, Da and Chen, Ruizhao and Cheng, Xu and Wu, Mengyue and Zhang, Quanshi},
  year={2021},
  booktitle = {The AAAI Conference on Artificial Intelligence (AAAI)}
}
```







