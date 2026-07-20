import numpy as np
from collections import Counter
from sklearn.base import clone

class RandomForest:
    def __init__(self,
                 base_estimator,
                 n_estimators=100,
                 max_samples=None,
                 max_features=None):
        """
        :param base_estimator: 基学习器
        :param n_estimators: 学习器的数量
        :param max_samples: 样本的采样比例或采样值
        :param max_features: 样本特征的采样比例
        """
        self.base_estimator=base_estimator
        self.n_estimators=n_estimators
        self.max_samples=max_samples
        self.max_features=max_features
        # 存储训练好的模型和对应的特征索引
        self.estimators_ = []
        self.feature_indices_list_ = []
    def _get_max_samples(self, n_samples):
        """
        计算每棵树的采样数量
        :param n_samples:一共的样本数量
        :return:每个弱学习器的采样数
        """
        if self.max_samples is None:
            return n_samples
        if isinstance(self.max_samples, float):
            return int(self.max_samples * n_samples)
        return min(self.max_samples, n_samples)
    def _get_max_features(self, n_features):
        """
        计算每棵树的特征数量
        :param n_features:
        :return:
        """
        if self.max_features is None:
            return n_features
        if isinstance(self.max_features, int):
            return min(self.max_features, n_features)
        if self.max_features == 'sqrt':
            return int(np.sqrt(n_features))
        if self.max_features == 'log2':
            return int(np.log2(n_features))
        if isinstance(self.max_features, float):
            return int(self.max_features * n_features)
    def fit(self,x,y):
        """
        训练随机森林
        :param x: 训练特征数据集
        :param y: 训练集标签
        """
        x = np.array(x)
        y = np.array(y)
        #获取样本数和特征数
        n_samples, n_features = x.shape
        #每个基学习器的样本数和特征数
        n_samples_base = self._get_max_samples(n_samples)
        n_features_base = self._get_max_features(n_features)
        original_feature_names = self.base_estimator.feature_names
        for i in range(self.n_estimators):
            #随机选择m条数据
            sample_indices = np.random.choice(n_samples, n_samples_base, replace=True)
            x_sample = x[sample_indices]
            y_sample = y[sample_indices]
            #随机选择k个特征
            feature_indices = np.random.choice(n_features, n_features_base, replace=True)
            x_selected = x_sample[:, feature_indices]
            #克隆一个树，每次使用完全一样的树
            if x_selected.ndim == 1:
                x_selected = x_selected.reshape(-1, 1)

            tree = clone(self.base_estimator)
            tree.feature_names = [original_feature_names[j] for j in feature_indices]
            tree.fit(x_selected, y_sample)
            #把当前训练好的树和特征索引存储
            self.estimators_.append(tree)
            self.feature_indices_list_.append(feature_indices)
        return self

    def predict(self, x):
        x = np.array(x)
        n_samples = x.shape[0]
        # 收集所有树的预测结果
        predictions = np.zeros((n_samples, self.n_estimators), dtype=object)
        for i, (tree, feature_indices) in enumerate(zip(self.estimators_,
                                                        self.feature_indices_list_)):
            x_selected = x[:, feature_indices]
            if x_selected.ndim == 1:
                x_selected = x_selected.reshape(-1, 1)
            predictions[:, i] = tree.predict(x_selected)
        # 平权投票
        final_predictions = np.zeros(n_samples, dtype=object)
        for i in range(n_samples):
            votes = Counter(predictions[i])
            final_predictions[i] = votes.most_common(1)[0][0]

        return final_predictions
if __name__ == "__main__":
    from CART import CART
    import numpy as np
    from sklearn.metrics import accuracy_score
    X_train = np.array([
        [25, '男', 5000, '本科'],
        [30, '女', 8000, '硕士'],
        [22, '男', 3000, '大专'],
        [35, '女', 12000, '博士'],
        [28, '男', 6000, '本科'],
        [32, '女', 9000, '硕士'],
        [20, '男', 2000, '高中'],
        [40, '女', 15000, '博士'],
        [26, '男', 4500, '本科'],
        [33, '女', 10000, '硕士']
    ], dtype=object)
    X_test = np.array([
        [28, '男', 4500, '本科'],
        [35, '女', 13000, '博士'],
        [22, '男', 2500, '大专'],
        [30, '女', 8500, '硕士']
    ], dtype=object)
    y_train = np.array(['否', '是', '否', '是', '否',
                        '是', '否', '是', '否', '是'])
    y_test = np.array(['否', '是', '否', '是'])
    feature_names = ['年龄', '性别', '收入', '学历']
    model = CART(feature_names)
    rf =RandomForest(base_estimator=model,
                 n_estimators=200,
                 max_samples=0.3,
                 max_features=0.3)
    rf.fit(X_train,y_train)
    pre=rf.predict(X_test)
    acc = accuracy_score(y_test, pre)
    print(acc)