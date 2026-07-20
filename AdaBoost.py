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
class cart_adaboost(object):
    def __init__(self, feature_names, feature_indices, min_samples_split=10, max_depth=None):
        self.root = None
        self.feature_names = feature_names
        self.min_samples_split = min_samples_split  # 当小于最小样本分割数的时候，不再生成树
        self.max_depth = max_depth  # 最大深度
        self.feature_indices = feature_indices  # 特征编号
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
    def weighted_majority(self, y, w):
        """加权众数"""
        unique_classes = np.unique(y)
        class_weight = {}
        for cls in unique_classes:
            class_weight[cls] = np.sum(w[y == cls])
        return max(class_weight, key=class_weight.get)
    def tree(self, x, y,w,depth=0):
        """
        生成当前树
        :param x: 当前剩余的特征矩阵
        :param y: 与x对应的y
        :param w: 当前权重
        :param depth: 当前深度
        """
        x = np.array(x, dtype=object)
        y = np.array(y)
        if (len(np.unique(y)) == 1 or#所有标签都相同
                self.is_same_on_features(x) or#所有特征取值相同
                len(y) < self.min_samples_split or# 节点样本数少于阈值
                (self.max_depth is not None and depth >= self.max_depth)):# 达到最大深度
            return TreeNode(label=self.weighted_majority(y, w))
        best_error = float('inf')
        best_feature_idx = None #特征的idx
        best_split_info = None  # 分割值
        best_left_mask = None  # 左侧树使用的剩余sample
        best_right_mask = None  # 右侧树使用的剩余sample
        for idx in self.feature_indices:
            x_col = x[:, idx]
            # 获取 x和y从小到大排序的索引
            idx_x = np.argsort(x_col)
            x_sorted = x_col[idx_x]
            y_sorted = y[idx_x]
            w_sorted=w[idx_x]
            averages = (x_sorted[:-1] + x_sorted[1:]) / 2  # 相邻两个元素的平均值组成的list
            for threshold in averages:
                left_mask = x_sorted <= threshold
                right_mask = x_sorted > threshold
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue  # 跳过无效切分
                    # 左侧预测为 +1，右侧预测为 -1
                pre = np.ones_like(y_sorted)
                pre[right_mask] = -1
                error = np.sum(w_sorted * (pre != y_sorted))
                if error < best_error:
                    best_error = error
                    best_split_info = threshold
                    best_feature_idx = idx
                    best_left_mask = x_col <= threshold
                    best_right_mask = x_col > threshold
        if best_feature_idx is None:
            return TreeNode(label=self.weighted_majority(y, w))
        node = TreeNode(
            feature_idx=best_feature_idx,
            feature_name=self.feature_names[best_feature_idx],
            split_info=best_split_info
        )
        if np.sum(best_left_mask) > 0:
            node.left = self.tree(x[best_left_mask], y[best_left_mask], w[best_left_mask],depth + 1)

        # 构建右子树
        if np.sum(best_right_mask) > 0:
            node.right = self.tree(x[best_right_mask], y[best_right_mask], w[best_right_mask],depth + 1)

        return node
    def fit(self, x, y,w):
        """
        训练模型
        """
        self.root = self.tree(x, y,w)
        return self
    def predict_sample(self, sample, node=None):
        """
        预测单个样本
        :param node: 当前预测节点
        :param sample:单个样本
        :return:叶子节点的label
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
    def predict(self, x):
        """
        批量预测
        :param x: 特征矩阵
        :return: 预测的类别数组
        """
        x = np.array(x, dtype=object)
        return np.array([self.predict_sample(sample) for sample in x])
class Adaboost(object):
    def __init__(self,model,n_estimators):
        """
        :param model: adaboost里面使用的模型
        :param n_estimators: 一共多少个模型
        """
        self.model=model
        self.n_estimators=n_estimators
        self.models = []  # 存储每棵树
        self.alphas = []  # 存储每棵树的话语权
    def fit(self,x,y):
        """
        构建adaboost模型
        :param x: 导入的全部特征
        :param y: 导入的全部label
        """
        x = np.array(x, dtype=object)
        y = np.array(y)
        #初始化
        n = x.shape[0]
        w = np.ones(n) / n
        for i in range(self.n_estimators):
            tree = cart_adaboost(
                feature_names=self.model.feature_names,
                feature_indices=self.model.feature_indices,
                min_samples_split=self.model.min_samples_split,
                max_depth=self.model.max_depth
            )
            tree.fit(x, y, w)
            y_pre = tree.predict(x)
            error = np.sum(w * (y_pre != y))
            error = max(error, 1e-10)  # 防止 error = 0
            error = min(error, 1 - 1e-10)  # 防止 error = 1
            #计算话语权
            alpha = 0.5 * np.log((1 - error) / error)
            #更新w
            w = w * np.exp(-alpha * y * y_pre)
            w = w / np.sum(w)  # 归一化
            #保存
            self.models.append(tree)
            self.alphas.append(alpha)

        return self
    def predict(self, x):
        """
        预测
        """
        x = np.array(x, dtype=object)
        # 收集所有树的预测结果
        pres = np.array([model.predict(x) for model in self.models])
        # 加权投票
        weighted_pres = np.dot(self.alphas, pres)
        return np.sign(weighted_pres)
if __name__ == "__main__":
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
    ], dtype=np.float64)
    y_train = np.array([-1, -1, -1, 1, -1, 1, -1, 1, -1, 1])
    X_test = np.array([
        [28, 4500, 4, 1],
        [35, 13000, 13, 3],
        [22, 2500, 2, 0],
        [30, 8500, 8, 2],
        [38, 11000, 11, 3]
    ], dtype=np.float64)
    y_test = np.array([-1, 1, -1, -1, 1])
    feature_names = ['年龄', '收入', '工龄', '学历等级']
    feature_indices = [0, 1, 2, 3]
    single_tree = cart_adaboost(
        feature_names=feature_names,
        feature_indices=feature_indices,
        min_samples_split=2,
        max_depth=3
    )
    w_uniform = np.ones(len(X_train)) / len(X_train)
    single_tree.fit(X_train, y_train, w_uniform)
    y_pre_single = single_tree.predict(X_test)
    acc_single = np.mean(y_pre_single == y_test)
    print(f"单棵树预测: {y_pre_single}")
    print(f"单棵树准确率: {acc_single:.4f}")
    base_tree = cart_adaboost(
        feature_names=feature_names,
        feature_indices=feature_indices,
        min_samples_split=2,
        max_depth=3
    )
    ada = Adaboost(model=base_tree, n_estimators=20)
    ada.fit(X_train, y_train)
    y_pre_ada = ada.predict(X_test)
    acc_ada = np.mean(y_pre_ada == y_test)
    print(f"AdaBoost 预测: {y_pre_ada}")
    print(f"AdaBoost 准确率: {acc_ada:.4f}")