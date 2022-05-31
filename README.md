# filter-pruning-with-weight-attention

pruning cifar10 :  pruning_cifar10.py

pruning imagenet: pruning_imagenet.py

参数设置可参考：https://github.com/he-y/filter-pruning-geometric-median
本文代码只是在该代码上修改了对应的剪枝算法。主要修改点集中在：def get_filter_similar(self, weight_torch, compress_rate, distance_rate, length, dist_type="l2"):方法中。我们使用权重注意力机制
选择冗余特征进行剪裁。
