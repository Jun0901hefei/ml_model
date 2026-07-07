import numpy as np
class TreeNode(object):
    def __init__(self,
                 feature_idx=None,#特征索引
                 feature_name=None,#特征名字
                 left=None,#左节点
                 right=None,#右节点
                 label=None,#输出标签
                 split_info=None):#划分值
        self.feature_idx=feature_idx
        self.feature_name = feature_name
        self.left = left
        self.right = right
        self.label = label
        self.split_info = split_info
    def is_leaf(self):
        """
        判断是否为叶子节点
        如果是叶子节点，返回True
        """
        return self.label is not None
class CART_Re(object):
    def __init__(self, feature_names,feature_indices,min_samples_split=10,max_depth=None):
        self.root = None
        self.feature_names = feature_names
        self.min_samples_split=min_samples_split#当小于最小样本分割数的时候，不再生成树
        self.max_depth=max_depth#最大深度
        self.feature_indices=feature_indices#特征编号
    def Least_Squares(self,y):
        """
        计算最小二乘
        :param y:标签
        :return: 当前情况下最小二乘数值
        """
        if len(y) == 0:
            return 0
        y_average=np.mean(y)
        ls=np.sum((y-y_average)**2)
        return ls
    def is_same_on_features(self, x):
        """
        判断所有特征列取值是不是一样
        :param x: 当前节点下所有特征列
        """
        for idx in self.feature_indices:
            # 检查该特征列是否只有一个唯一值
            if len(np.unique(x[:, idx])) > 1:
                return False  # 只要有一个特征有不同取值，就返回False
        return True  # 所有特征都只有唯一值
    def Cart_re(self,x, y,depth=0):
        """
        生成CART回归树
        :param depth: 当前深度
        :param x: 当前剩余的特征矩阵
        :param y: 与x对应的y
        """
        x = np.array(x, dtype=object)
        y = np.array(y)
        if (len(np.unique(y)) == 1 or#所有标签都相同
                self.is_same_on_features(x) or#所有特征取值相同
                len(y) < self.min_samples_split or# 节点样本数少于阈值
                (self.max_depth is not None and depth >= self.max_depth)):# 达到最大深度
            return TreeNode(label=np.mean(y))

        best_ls = float('inf')
        best_feature_idx = None
        best_split_info = None  #分割值
        best_left_mask = None   #左侧树使用的剩余sample
        best_right_mask = None  #右侧树使用的剩余sample
        for idx in self.feature_indices:
            x_col = x[:, idx]
            #获取 x和y从小到大排序的索引
            idx_x = np.argsort(x_col)
            x_sorted = x_col[idx_x]
            y_sorted = y[idx_x]
            averages = (x_sorted[:-1] + x_sorted[1:]) / 2#相邻两个元素的平均值组成的list
            for i in averages:
                if np.sum(x_sorted < i) == 0 or np.sum(x_sorted > i) == 0:
                    continue  # 跳过无效切分
                less_than_i = y_sorted[x_sorted <= i]
                greater_than_i = y_sorted[x_sorted > i]
                less_ls=self.Least_Squares(less_than_i)
                great_ls = self.Least_Squares(greater_than_i)
                ls=less_ls+great_ls
                if ls<best_ls:
                    best_feature_idx=idx
                    best_split_info = i
                    best_left_mask = x_col <= i
                    best_right_mask = x_col > i
        if best_feature_idx is None:
            return TreeNode(label=np.mean(y))
        node = TreeNode(
            feature_idx=best_feature_idx,
            feature_name=self.feature_names[best_feature_idx],
            split_info=best_split_info
        )
        if np.sum(best_left_mask) > 0:
            node.left = self.Cart_re(x[best_left_mask], y[best_left_mask], depth + 1)

        # 构建右子树
        if np.sum(best_right_mask) > 0:
            node.right = self.Cart_re(x[best_right_mask], y[best_right_mask],depth + 1)

        return node
    def fit(self, x, y):
        """
        训练模型
        """
        self.root = self.Cart_re(x, y)
        return self
    def predict_sample(self, sample, node=None):
        """
        预测单个样本
        :param node: 当前预测节点
        :param sample:单个样本
        :return:叶子节点的lable
        """
        if node is None:
            node = self.root
        if node.is_leaf():
            return node.label
        feature_value = sample[node.feature_idx]
        if feature_value <= node.split_info:
            return self.predict_sample(sample, node.left)
        else:
            return self.predict_sample(sample, node.right)
    def predict(self, X):
        """
        批量预测
        :param X: 特征矩阵
        :return: 预测的类别数组
        """
        X = np.array(X, dtype=object)
        return np.array([self.predict_sample(sample) for sample in X])
if __name__ == "__main__":
    """
    以下是测试代码
    """
    import numpy as np
    X_train = np.array([
        [25, 5000, 5, 1],
        [30, 8000, 8, 2],
        [22, 3000, 3, 0],
        [35, 12000, 12, 3],
        [28, 6000, 6, 1],
        [32, 9000, 9, 2],
        [20, 2000, 2, 0],
        [40, 15000, 15, 3],
        [26, 4500, 4, 1],
        [33, 10000, 10, 2]
    ], dtype=object)
    X_test = np.array([
        [28, 4500, 4, 1],
        [35, 13000, 13, 3],
        [22, 2500, 2, 0],
        [30, 8500, 8, 2]
    ], dtype=object)
    y_train = np.array([12, 20, 8, 35, 15, 22, 5, 40, 11, 25])
    y_test = np.array([13, 38, 7, 24])
    feature_names = ['年龄', '收入', '工龄', '学历等级']
    feature_indices = [0, 1, 2, 3]
    model = CART_Re(feature_names, feature_indices, min_samples_split=3, max_depth=4)
    model.fit(X_train, y_train)
    y_pre = model.predict(X_test)
    print(y_test)
    print(y_pre)

