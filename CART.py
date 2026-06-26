import numpy as np
from collections import Counter
class TreeNode:
    def __init__(self,
                 feature_idx=None,
                 feature_name=None,
                 split_type=None,  # 'num' 或 'char'
                 split_info=None,  # 数值：分割值；离散：左值
                 left=None,
                 right=None,
                 label=None):
        self.feature_idx = feature_idx
        self.feature_name = feature_name
        self.split_type = split_type
        self.split_info = split_info
        self.left = left
        self.right = right
        self.label = label
    def is_leaf(self):
        """
        判断是否为叶子节点
        如果是叶子节点，返回True
        """
        return self.label is not None

class CART(object):
    def __init__(self, feature_names):
        self.root = None
        self.feature_names = feature_names
    def Gini(self,y):
        """
        计算基尼值
        :param y: 当前节点的标签
        :return: 基尼值
        """
        if len(y) == 0:
            return 0
        class_counts = Counter(y)
        probs = np.array([count / len(y) for count in class_counts.values()])
        return 1-np.sum(probs*probs)
    def Gini_rate_char(self,x_col,y):
        """
        计算非数值基尼系数
        :param x_col: 特征列
        :param y: 和特征列对应的标签
        :return: 基尼系数
        """
        x_col = np.array(x_col, dtype=object)
        y = np.array(y)
        unique_values = np.unique(x_col)
        #如果是多类
        if len(unique_values)>2:
            best_gini = float('inf')
            best_left_val = None

            for value in unique_values:
                y_sub = y[x_col == value]
                y_other_sub = y[x_col != value]

                gini = (self.Gini(y_sub) * (len(y_sub) / len(y)) +
                        self.Gini(y_other_sub) * (len(y_other_sub) / len(y)))

                if gini < best_gini:
                    best_gini = gini
                    best_left_val = value

            return best_left_val, best_gini
        #如果是一类
        elif len(unique_values)==1:
            return None, float('inf')
        #如果刚好两类
        elif len(unique_values)==2:
            Gini_rate=0
            for value in unique_values:
                y_sub = y[x_col == value]
                Gini_rate+=self.Gini(y_sub)*(len(y_sub)/len(y))
            return unique_values[0], Gini_rate

    def Gini_rate_num(self,x_col,y):
        """
        计算数值基尼系数
        :param x_col: 特征列
        :param y: 和特征列对应的标签
        :return: 基尼系数和对应的分类节点
        """
        x_col = np.array(x_col, dtype=float)
        y = np.array(y)
        #获得排序之后的二维列表
        data=np.array([x_col,y],dtype=object)#避免强制转换数字为字符串
        data_indices=np.argsort(data[0])
        data_sort=data[:,data_indices]
        data_sort=data_sort.T
        #获取特征列和标签列
        x=data_sort[:,0]
        y=data_sort[:, 1]
        #获取中间值的列表
        mid_values = [(x[i] + x[i + 1]) / 2 for i in range(len(x) - 1)]
        if not mid_values:
            return None, float('inf')
        Gini_rate = dict()
        #找出最小的基尼系数的分类和值
        for value in mid_values:
            y_sub=y[x<value]
            y_other_sub=y[x>value]
            Gini_rate[value] = self.Gini(y_sub) * (len(y_sub) / len(y)) + self.Gini(y_other_sub) * (len(y_other_sub) / len(y))
        return min(Gini_rate.items(), key=lambda x: x[1])
    def is_same_on_features(self, x, feature_indices):
        """
        判断所有特征列取值是不是一样
        :param x: 当前节点下所有特征列
        :param feature_indices: 当前节点下的所有特征序列
        """
        for idx in feature_indices:
            # 检查该特征列是否只有一个唯一值
            if len(np.unique(x[:, idx])) > 1:
                return False  # 只要有一个特征有不同取值，就返回False
        return True  # 所有特征都只有唯一值
    def CART_tree(self, x, y,feature_indices):
        """
        生成cart树
        :param x: 当前剩余的特征矩阵
        :param y: 与x对应的y
        :param feature_indices: 与x对应的特征序列
        """
        x = np.array(x, dtype=object)
        y = np.array(y)
        # 如果所有样本属于同一类别，返回叶子节点
        if len(np.unique(y)) == 1:
            return TreeNode(label=y[0])
        # 如果没有剩余特征，或者所有特征取值相同返回多数类别
        if len(feature_indices) == 0 or self.is_same_on_features(x, feature_indices):
            return TreeNode(label=Counter(y).most_common(1)[0][0])

        best_gini = float('inf')
        best_feature_idx = None
        best_split_info = None  # 数值：分割值；离散：左值
        best_split_type = None  # 'num' 或 'char'
        best_left_mask = None
        best_right_mask = None
        for idx in feature_indices:
            x_col = x[:, idx]
            #数值特征
            if np.issubdtype(x_col.dtype, np.number):
                split_value, gini = self.Gini_rate_num(x_col, y)
                if gini < best_gini:
                    best_gini = gini
                    best_feature_idx = idx
                    best_split_info = split_value
                    best_split_type = 'num'
                    best_left_mask = x_col <= split_value
                    best_right_mask = x_col > split_value
            else:
                # 离散特征
                left_val, gini = self.Gini_rate_char(x_col, y)
                if gini < best_gini:
                    best_gini = gini
                    best_feature_idx = idx
                    best_split_info = left_val
                    best_split_type = 'char'
                    best_left_mask = (x_col == left_val)
                    best_right_mask = (x_col != left_val)
        node = TreeNode(
            feature_idx=best_feature_idx,
            feature_name=self.feature_names[best_feature_idx],
            split_type=best_split_type,
            split_info=best_split_info
        )
        remaining_indices = [i for i in feature_indices if i != best_feature_idx]
        majority_label = Counter(y).most_common(1)[0][0]
        #构建左子树
        if np.sum(best_left_mask) > 0:
            node.left = self.CART_tree(x[best_left_mask], y[best_left_mask], remaining_indices)
        else:
            node.left = TreeNode(label=majority_label)#防御性编程，可以省略
        # 构建右子树
        if np.sum(best_right_mask) > 0:
            node.right = self.CART_tree(x[best_right_mask], y[best_right_mask], remaining_indices)
        else:
            node.right = TreeNode(label=majority_label)#防御性编程，可以省略
        return node
    def fit(self, x, y):
        """
        训练模型
        """
        self.root = self.CART_tree(x, y, list(range(x.shape[1])))
        return self
    def predict_sample(self,sample,node=None):
        """
        预测单个样本
        :param node: 当前预测节点
        :param sample:单个样本
        :return:
        """
        if node is None:
            node = self.root
        if node.is_leaf():
            return node.label
        feature_value = sample[node.feature_idx]
        if node.split_type == 'num':#如果是数值类型
            if feature_value <= node.split_info:
                return self.predict_sample(sample, node.left)
            else:
                return self.predict_sample(sample, node.right)
        else:#如果是字符类型
            if feature_value == node.split_info:
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
    import numpy as np

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
    model=CART(feature_names)
    model.fit(X_train,y_train)
    y_pre=model.predict(X_test)
    print(y_test)
    print(y_pre)
    """
    本仓库给出了主流机器学习的模型代码
包括：KNN，线性回归——正规方程法，线性回归——minibatch梯度下降法，线性回归——全梯度下降，id3决策树，c4.5决策树，CART决策树"""