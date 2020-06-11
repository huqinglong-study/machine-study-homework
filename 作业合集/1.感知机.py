
###原始形式

import pandas as pd
import numpy as np
import cv2
import random
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class Perceptron(object):

    def __init__(self):
        self.learning_step = 0.00001
        self.max_iteration = 5000

    def predict_(self, x):
        wx = sum([self.w[j] * x[j] for j in range(len(self.w))])
        return int(wx > 0)

    def train(self, features, labels):
        self.w = [0.0] * (len(features[0]) + 1)

        correct_count = 0
        time = 0
        

        while time < self.max_iteration:
            index = random.randint(0, len(labels) - 1)
            x = list(features[index])
            x.append(1.0)
            y = 2 * labels[index] - 1
            wx = sum([self.w[j] * x[j] for j in range(len(self.w))])

            if wx * y > 0:
                correct_count += 1
                if correct_count > self.max_iteration:
                    break
                continue

            for i in range(len(self.w)):
                self.w[i] += self.learning_step * (y * x[i])

    def predict(self,features):
        labels = []
        for feature in features:
            x = list(feature)
            x.append(1)
            labels.append(self.predict_(x))
        return labels


if __name__ == '__main__':

    print ('Start read data')

    time_1 = time.time()
    #将数据集读入
    raw_data = pd.read_csv('E:/19人工智能实践and20机器学习/机器学习/lihang_book_algorithm-master/lihang_book_algorithm-master/data/train_binary.csv', header=0)
    data = raw_data.values

    imgs = data[0::, 1::]
    labels = data[::, 0]

    # 选取 2/3 数据作为训练集， 1/3 数据作为测试集
    train_features, test_features, train_labels, test_labels = train_test_split(
        imgs, labels, test_size=0.33, random_state=23323)
    # print train_features.shape
    # print train_features.shape

    time_2 = time.time()
    print ('read data cost ', time_2 - time_1, ' second', '\n')

    print ('Start training')
    p = Perceptron()
    p.train(train_features, train_labels)

    time_3 = time.time()
    print ('training cost ', time_3 - time_2, ' second', '\n')

    print ('Start predicting')
    test_predict = p.predict(test_features)
    time_4 = time.time()
    print ('predicting cost ', time_4 - time_3, ' second', '\n')

    score = accuracy_score(test_labels, test_predict)
    print ("The accruacy socre is ", score)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:

##对偶形式

import pandas as pd
import numpy as np
import cv2
import random
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class Perceptron(object):

    def __init__(self):
        self.learning_step = 0.00001
        self.max_iteration = 5000

    def predict_(self, x):
        wx = sum([self.w[j] * x[j] for j in range(len(self.w))])
        return int(wx > 0)
    
    #训练
    def train(self, features, labels):
        self.a = [0.0] * len(features)  #定义alpha向量
        
        #构建gram矩阵
        gram=np.dot(features,features.T)
        
        correct_count = 0    #记录纠正次数
        time = 0
        

        while time < self.max_iteration:
            index = random.randint(0, len(labels) - 1)
            x = list(features[index])
            x.append(1.0)
            y = 2 * labels[index] - 1
            labelstran= 2 * labels -1   #将labels转化为1和-1
            
            ##wx = sum([self.w[j] * x[j] for j in range(len(self.w))])  原始形式的Omega
            
            wx = np.sum(self.a*labelstran*gram[index])  #对偶形式的Omega

            if wx * y > 0:
                correct_count += 1
                if correct_count > self.max_iteration:
                    break
                continue

            #for i in range(len(self.w)):
            self.a[index] += self.learning_step   #对偶形式的学习
        self.w = np.sum(self.a * labelstran * features.T,axis=1)  #计算Omega
            

    def predict(self,features):
        labels = []
        for feature in features:
            x = list(feature)
            x.append(1)
            labels.append(self.predict_(x))
        return labels


if __name__ == '__main__':

    print ('Start read data')

    time_1 = time.time()

    raw_data = pd.read_csv('E:/19人工智能实践and20机器学习/机器学习/github/machine-study/train_binary.csv', header=0)
    data = raw_data.values

    imgs = data[0::, 1::]
    labels = data[::, 0]

    # 选取 2/3 数据作为训练集， 1/3 数据作为测试集
    train_features, test_features, train_labels, test_labels = train_test_split(
        imgs, labels, test_size=0.33, random_state=23323)
    # print train_features.shape
    # print train_features.shape

    time_2 = time.time()
    print ('read data cost ', time_2 - time_1, ' second', '\n')

    print ('Start training')
    p = Perceptron()
    p.train(train_features, train_labels)

    time_3 = time.time()
    print ('training cost ', time_3 - time_2, ' second', '\n')

    print ('Start predicting')
    test_predict = p.predict(test_features)
    time_4 = time.time()
    print ('predicting cost ', time_4 - time_3, ' second', '\n')

    score = accuracy_score(test_labels, test_predict)
    print ("The accruacy socre is ", score)


# In[ ]:




