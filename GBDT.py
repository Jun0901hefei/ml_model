import copy
import numpy as np
class GBDT(object):
    def __init__(self,
                 cart_tree,
                 n_estimators,
                 learning_rate):
        """
        :param cart_tree: 导入模型实例
        :param n_estimators: 树的数量
        :param learning_rate: 学习率
        """
        self.cart_tree=cart_tree
        self.n_estimators=n_estimators
        self.learning_rate=learning_rate
        #储存训练好的模型
        self.trees= []
        # 初始预测值
        self.initial_pre= None

    def fit(self,x,y):
        """
        训练多个cart_re组成的GBDT
        """
        x = np.array(x)
        y = np.array(y)
        #初始化预测值
        self.initial_pre=np.mean(y)
        predictions = np.full(len(y), self.initial_pre)
        #计算每个样本的残差
        residuals = y - predictions
        for i in range(self.n_estimators):
            tree = copy.deepcopy(self.cart_tree)
            tree.fit(x, residuals)
            tree_outputs = tree.predict(x)
            # 更新预测值：加上学习率 × 树的输出
            predictions += self.learning_rate * tree_outputs
            # 更新残差：y - 新预测值
            residuals = y - predictions
            # 保存这棵树
            self.trees.append(tree)
        return self
    def predict(self, x):
        x = np.array(x)
        predictions = np.full(len(x), self.initial_pre)

        for tree in self.trees:
            predictions += self.learning_rate * tree.predict(x)

        return predictions

if __name__ == "__main__":
    """
    以下是测试代码
    """
    import numpy as np
    from CART_re import CART_Re
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
    GBDT_model=GBDT(cart_tree=model,
                 n_estimators=100,
                 learning_rate=0.1)
    GBDT_model.fit(X_train,y_train)
    y_pre=GBDT_model.predict(X_test)
    print(y_test)
    print(y_pre)