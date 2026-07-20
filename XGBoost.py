import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp

class TreeNode(object):
    def __init__(self,
                 feature_idx=None,  # 特征索引
                 left=None,  # 左节点
                 right=None,  # 右节点
                 label=None,  # 输出标签
                 split_info=None):  # 划分值
        self.feature_idx = feature_idx
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
class DecisionTree:
    def __init__(self,
                 max_depth=3,
                 min_samples_split=2,
                 lambda_=0.5,
                 gamma=0.5,
                 n_jobs=1
                ):
        self.max_depth = max_depth#最大深度
        self.min_samples_split = min_samples_split#最小分割的样本数
        self.lambda_=lambda_#正则化项中节点个数的参数
        self.gamma=gamma#正则化项中叶子节点输出值组成向量的参数
        self.root = None
        self.n_jobs = n_jobs

    def _obj(self,G,H):
        """
        当前节点的obj函数
        :param G: 当前节点所有样本loss函数一阶导的和
        :param H: 当前节点所有样本loss函数二阶导的和
        :return:当前节点的obj值
        """
        obj_value = -0.5 * G ** 2 / (H + self.lambda_) + self.gamma
        return obj_value
    def _gain(self,G_l,G_r,H_l,H_r):
        """
        计算出gain函数的值
        :param G_l: 分割后左节点的样本的loss函数一阶导的和
        :param G_r: 分割后右节点的样本的loss函数一阶导的和
        :param H_l: 分割后左节点的样本的loss函数二阶导的和
        :param H_r: 分割后右节点的样本的loss函数二阶导的和
        :return: gain函数的值
        """
        return 0.5*(G_l ** 2/(H_l+self.lambda_) +
                    G_r ** 2/(H_r+self.lambda_) -
                    (G_l+G_r) ** 2/(H_l+H_r+self.lambda_))-self.gamma
    def is_same_on_features(self, x,features_idx):
        """
        判断所有特征列取值是不是一样
        :param features_idx: 特征序号列表
        :param x: 当前节点下所有特征列
        """
        for idx in features_idx:
            # 检查该特征列是否只有一个唯一值
            if len(np.unique(x[:, idx])) > 1:
                return False  # 只要有一个特征有不同取值，就返回False
        return True  # 所有特征都只有唯一值
    def tree(self,x, y,g,h,depth=0):
        """
        构建一棵树
        :param x:当前剩余的特征矩阵
        :param y:与x对应的y
        :param g:当前节点所有样本的loss一阶导数list
        :param h:当前节点所有样本的loss二阶导数list
        :param depth:当前深度
        """
        x= np.array(x)
        y = np.array(y)
        g = np.array(g, dtype=float)
        h = np.array(h, dtype=float)
        if len(y) == 0:
            return TreeNode(label=0.0)
        features_idx=[i for i in range(x.shape[1])]
        # 当前节点的所有样本的loss一阶导之和
        G = np.sum(g)
        # 当前节点的所有样本的loss二阶导之和
        H = np.sum(h)
        if H + self.lambda_ == 0:
            leaf_weight = 0.0
        else:
            leaf_weight = -G / (H + self.lambda_)
        if np.isnan(leaf_weight) or np.isinf(leaf_weight):
            leaf_weight = 0.0
        if np.any(np.isnan(g)) or np.any(np.isinf(g)):
            return TreeNode(label=leaf_weight)
        if np.any(np.isnan(h)) or np.any(np.isinf(h)):
            return TreeNode(label=leaf_weight)

            #如果梯度全为 0，返回叶子节点
        if np.allclose(g, 0, atol=1e-8):
            return TreeNode(label=leaf_weight)
        #预剪枝
        if (len(np.unique(y)) == 1 or#所有标签都相同
                self.is_same_on_features(x,features_idx) or#所有特征取值相同
                len(y) < self.min_samples_split or# 节点样本数少于阈值
                (self.max_depth is not None and depth >= self.max_depth)):# 达到最大深度
            return TreeNode(label=leaf_weight)
        best_gain=-float('inf')
        best_feature_idx = None #分割的特征序号
        best_split_info = None  # 分割值
        best_left_mask = None  # 左侧树使用的剩余sample
        best_right_mask = None  # 右侧树使用的剩余sample
        n_samples = x.shape[0]
        for idx in features_idx:
            x_col = x[:, idx]
            # 按特征值排序
            sorted_idx = np.argsort(x_col)
            x_sorted = x_col[sorted_idx]
            g_sorted = g[sorted_idx]
            h_sorted = h[sorted_idx]
            # 计算累积和
            G_total = np.sum(g_sorted)
            H_total = np.sum(h_sorted)
            #向量化计算左侧所有的G
            G_left_cum = np.cumsum(g_sorted[:-1])
            H_left_cum = np.cumsum(h_sorted[:-1])
            #计算所有可以分割点的索引
            unique_mask = np.concatenate([np.diff(x_sorted)> 1e-8, [False]])
            split_positions = np.where(unique_mask)[0]
            #当没有可以分割的点的时候直接跳出此次循环
            if len(split_positions) == 0:
                continue
            #找出左侧，右侧的G和H
            G_l = G_left_cum[split_positions]
            H_l = H_left_cum[split_positions]
            G_r = G_total - G_l
            H_r = H_total - H_l
            #获取所有分割点gain的数组
            gains = self._gain(G_l,G_r,H_l,H_r)
            #gain值最大的索引
            best_pos_local = np.argmax(gains)
            #最大的gain值
            max_gain = gains[best_pos_local]
            if max_gain > 0 and max_gain > best_gain:
                best_gain = max_gain
                best_feature_idx = idx
                #分割点的idx，x_sorted的idx
                split_pos = split_positions[best_pos_local]
                #最佳分割值
                best_split_info = (x_sorted[split_pos] + x_sorted[split_pos + 1]) / 2
                #初始化左侧mask
                best_left_mask = np.zeros(n_samples, dtype=bool)
                best_left_mask[sorted_idx[:split_pos + 1]] = True
                best_right_mask = ~best_left_mask
        node=TreeNode(
            feature_idx=best_feature_idx,
            split_info=best_split_info,
        )
        # 如果 Gain <= 0，不分裂，返回叶子节点
        if best_feature_idx is None or best_gain <= 0:
            leaf_weight = -G / (H + self.lambda_)
            return TreeNode(label=leaf_weight)
        if np.sum(best_left_mask) > 0:
            node.left = self.tree(
                x[best_left_mask],
                y[best_left_mask],
                g[best_left_mask],
                h[best_left_mask],
                depth + 1
            )
        if np.sum(best_right_mask) > 0:
            node.right = self.tree(
                x[best_right_mask],
                y[best_right_mask],
                g[best_right_mask],
                h[best_right_mask],
                depth + 1
            )
        return node
    def fit(self, x, y, g, h):
        """训练单棵树（外部传入 g 和 h）"""
        self.root = self.tree(x, y, g, h, depth=0)
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
        x = np.array(x)
        return np.array([self.predict_sample(sample) for sample in x])
class XGBOOST_re:
    def __init__(self,
                 num_rounds=100,
                 learning_rate=0.1,
                 loss='Least squares',
                 **tree_params
                 ):
        self.num_rounds = num_rounds
        self.learning_rate = learning_rate
        self.trees = []#储存训练好的树
        self.loss = loss
        self.tree_params = tree_params#储存树的参数
        self.y_mean=None#训练数据均值
    def _gi(self,y,y_pre_before):
        """
        计算每个样本的损失函数的一阶导的值
        :param y: label
        :param y_pre_before: 上一轮的预测值
        :return: 一阶导函数值
        """
        if self.loss=='Least squares':
            return 2*(y_pre_before-y)
        elif self.loss=="squared_log_error":
            return (np.log1p(y_pre_before) - np.log1p(y)) / (y_pre_before + 1)

    def _hi(self, y, y_pre_before):
        """
        计算每个样本的损失函数的二阶导的值
        :param y: label
        :param y_pre_before: 上一轮的预测值
        :return: 二阶导函数值
        """
        if self.loss == 'Least squares':
            return np.full_like(y, 2.0)
        elif self.loss == "squared_log_error":
            return 2 * (np.log1p(y) - np.log1p(y_pre_before) - 1) / (y_pre_before + 1) ** 2

    def fit(self, x, y, verbose=True):
        """
        训练xgboost，内置每棵树在训练的时候要传入的H和G，都是用的上一个树的y_pre算出来的
        :param x: 训练集
        :param y: 测试集
        :param verbose: 是否展示训练过程
        """
        x = np.array(x)
        y = np.array(y, dtype=float)
        #保存训练数据均值
        self.y_mean = np.mean(y)
        #用均值初始化先验预测值
        y_pre = np.full_like(y, self.y_mean, dtype=float)
        for t in range(self.num_rounds):
            g = self._gi(y, y_pre)
            h = self._hi(y, y_pre)
            tree = DecisionTree(**self.tree_params)
            tree.fit(x, y, g, h)
            self.trees.append(tree)
            y_pre += self.learning_rate * tree.predict(x)
            if verbose and (t + 1) % 10 == 0:
                mse = np.mean((y - y_pre) ** 2)
                print(f"轮次 {t + 1}/{self.num_rounds}, MSE: {mse:.6f}")
        return self

    def predict(self, x):
        """
        预测
        :param x: 测试集
        :return: 预测出的结果
        """
        x = np.array(x, dtype=object)
        y_pre = np.full(x.shape[0], self.y_mean, dtype=float)
        for tree in self.trees:
            y_pre += self.learning_rate * tree.predict(x)

        return y_pre
class XGBOOST_multi:
    def __init__(self,
                 num_rounds=100,
                 learning_rate=0.1,
                 **tree_params
                 ):
        self.num_rounds = num_rounds
        self.learning_rate = learning_rate
        self.tree_params = tree_params
        self.trees = []  # 每轮存储 K 棵树
        self.num_classes = None

    def _softmax(self, logits):
        """
        Softmax 将 logits 转为概率
        :param logits: 所有样本，所有类别的分数矩阵
        :return: 所有样本，所有类别的概率矩阵
        """
        #防止数值溢出，将最大值规定为0
        logits_shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits_shifted)
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    def fit(self, x, y, verbose=True):
        """
        训练xgboost，内置每棵树在训练的时候要传入的H和G，都是用的上一个树的probs算出来的
        :param x: 训练集
        :param y: 测试集
        :param verbose: 是否展示训练过程
        :return:
        """
        x = np.array(x)
        y = np.array(y, dtype=int)
        #样本数
        n_samples = x.shape[0]
        # 自动检测类别数
        self.num_classes = len(np.unique(y))
        K = self.num_classes
        # One-hot 编码
        y_one_hot = np.eye(K)[y]
        # 初始化 分数 = 0
        logits = np.zeros((n_samples, K))
        for t in range(self.num_rounds):
            probs = self._softmax(logits)
            probs = np.clip(probs, 1e-15, 1 - 1e-15)
            # 存储第t轮训练的k个树
            trees_t = []
            for k in range(K):
                #第k个树所有样本loss的一阶导
                g_k = probs[:, k] - y_one_hot[:, k]
                # 第k个树所有样本loss的二阶导
                h_k = probs[:, k] * (1 - probs[:, k])
                if np.allclose(g_k, 0, atol=1e-8):
                    tree = DecisionTree(**self.tree_params)
                    tree.root = TreeNode(label=0.0)
                    trees_t.append(tree)
                    continue

                if np.any(np.isnan(g_k)) or np.any(np.isinf(g_k)):
                    tree = DecisionTree(**self.tree_params)
                    tree.root = TreeNode(label=0.0)
                    trees_t.append(tree)
                    continue
                #训练树
                tree = DecisionTree(**self.tree_params)
                tree.fit(x, y_one_hot[:, k], g_k, h_k)
                trees_t.append(tree)
                #更新当前y类别的分数
                logits[:, k] += self.learning_rate * tree.predict(x)
            self.trees.append(trees_t)
            if verbose and (t + 1) % 10 == 0:
                pre = np.argmax(self._softmax(logits), axis=1)
                acc = np.mean(pre == y)
                print(f"轮次 {t + 1}/{self.num_rounds}, 准确率: {acc:.4f}")
        return self

    def predict(self, x):
        x = np.array(x)
        n_samples = x.shape[0]
        K = self.num_classes
        #与fit保持一致都要初始化为0
        logits = np.zeros((n_samples, K))
        for trees_t in self.trees:
            for k, tree in enumerate(trees_t):
                logits[:, k] += self.learning_rate * tree.predict(x)
        probs = self._softmax(logits)
        return np.argmax(probs, axis=1)
if __name__ == "__main__":
    """
    以下是测试代码
    """
    #回归测试
    import numpy as np
    from sklearn.metrics import mean_squared_error
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
    model = XGBOOST_re()
    model.fit(X_train, y_train)
    y_pre = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pre)
    print(y_test)
    print(y_pre)
    print(mse)
# 简单多分类测试
    X = np.array([
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 5],
        [5, 6],
        [6, 7]
    ], dtype=object)
    y = np.array([0, 0, 1, 1, 2, 2])

    model = XGBOOST_multi(
        num_rounds=10,
        learning_rate=0.1,
        max_depth=2,
        min_samples_split=2
    )
    model.fit(X, y, verbose=True)
    print(model.predict(X))
