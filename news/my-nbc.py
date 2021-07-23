import os
import random
from sklearn.naive_bayes import *
import jieba
import matplotlib.pyplot as plt

def text_processing(folder_path, test_size = 0.2):
    folder_list = os.listdir(folder_path)           # 查看folder_path下的文件
    data_list = []                                  # 训练集
    class_list = []

    for folder in folder_list:
        son_folder_path = os.path.join(folder_path, folder)         # 根据子文件夹，生成新的路径
        files = os.listdir(son_folder_path)                         # 获取子文件夹下的文件

        j = 1
        for file in files:
            if j > 100:                                             # 每类txt样本数最多100个
                break
            with open(os.path.join(son_folder_path, file), 'r', encoding='utf-8') as f:
                raw = f.read()

            word_cut = jieba.cut(raw, cut_all=False)                # 精简模式，返回一个可迭代的generator
            word_list = list(word_cut)

            data_list.append(word_list)
            class_list.append(folder)
            j += 1

    data_class_list = list(zip(data_list, class_list))              # zip压缩合并，将数据和标签对应压缩
    random.shuffle(data_class_list)                                 # 将data_class_list乱序
    index = int(len(data_class_list) * test_size) + 1               # 训练集和测试集切分的索引值
    train_list = data_class_list[index:]                            # 训练集
    test_list = data_class_list[:index]                             # 测试集
    train_data_list, train_class_list = zip(*train_list)            # 训练集解压缩
    test_data_list, test_class_list = zip(*test_list)               # 测试集解压缩

    all_words_dict = {}                                             # 统计训练集词频
    for word_list in train_data_list:
        for word in word_list:
            if word in all_words_dict.keys():
                all_words_dict[word] += 1
            else:
                all_words_dict[word] = 1

    # 根据键的值倒序排序
    all_words_tuple_list = sorted(all_words_dict.items(), key = lambda x:x[1], reverse=True)
    all_words_list, all_words_nums = zip(*all_words_tuple_list)
    all_words_list = list(all_words_list)
    return all_words_list, train_data_list, test_data_list, train_class_list, test_class_list

def stopwords_set(words_file):
    words_set = set()
    with open(words_file, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            word = line.strip()
            if len(word) > 0:
                words_set.add(word)
    return words_set

def words_dict(all_words_list, deleteN, stopwords_set=None):
    if stopwords_set is None:
        stopwords_set = set()
    feature_words = []
    n = 1
    for t in range(deleteN, len(all_words_list), 1):
        if n > 1000:
            break
        if not all_words_list[t].isdigit() and all_words_list[t] not in stopwords_set and 1 < len(all_words_list[t]) < 5:
            feature_words.append(all_words_list[t])
        n += 1
    return feature_words

def text_features(train_data_list, test_data_list, feature_words):
    def text_features(text, feature_words):  # 出现在特征集中，则置1
        text_words = set(text)
        features = [1 if word in text_words else 0 for word in feature_words]
        return features

    train_feature_list = [text_features(text, feature_words) for text in train_data_list]
    test_feature_list = [text_features(text, feature_words) for text in test_data_list]
    return train_feature_list, test_feature_list

def text_MultinomialNB(train_feature_list, test_feature_list, train_class_list, test_class_list):
    classifier = MultinomialNB()
    classifier.fit(train_feature_list, train_class_list)
    test_accuracy = classifier.score(test_feature_list, test_class_list)
    return test_accuracy

def text_GaussianNB(train_feature_list, test_feature_list, train_class_list, test_class_list):
    classifier = GaussianNB()
    classifier.fit(train_feature_list, train_class_list)
    test_accuracy = classifier.score(test_feature_list, test_class_list)
    return test_accuracy

def text_CategoricalNB(train_feature_list, test_feature_list, train_class_list, test_class_list):
    classifier = CategoricalNB()
    classifier.fit(train_feature_list, train_class_list)
    test_accuracy = classifier.score(test_feature_list, test_class_list)
    return test_accuracy

def text_ComplementNB(train_feature_list, test_feature_list, train_class_list, test_class_list):
    classifier = ComplementNB()
    classifier.fit(train_feature_list, train_class_list)
    test_accuracy = classifier.score(test_feature_list, test_class_list)
    return test_accuracy

def text_BernoulliNB(train_feature_list, test_feature_list, train_class_list, test_class_list):
    classifier = BernoulliNB()
    classifier.fit(train_feature_list, train_class_list)
    test_accuracy = classifier.score(test_feature_list, test_class_list)
    return test_accuracy

if __name__ == '__main__':
    # 文本预处理
    folder_path = './SogouC/Sample'
    all_words_list, train_data_list, test_data_list, train_class_list, test_class_list = text_processing(folder_path, test_size=0.15)

    # 生成stopwords_set
    stopwords_file = './stopwords_cn.txt'
    stopwords_set = stopwords_set(stopwords_file)


    test_accuracy_list=[[],[],[],[],[]]
    deleteNs = range(0, 1000, 20)
    for deleteN in deleteNs:
        feature_words = words_dict(all_words_list, deleteN, stopwords_set)
        train_feature_list, test_feature_list = text_features(train_data_list, test_data_list, feature_words)
        test_accuracy = text_MultinomialNB(train_feature_list, test_feature_list, train_class_list, test_class_list)
        test_accuracy_list[0].append(test_accuracy)
        test_accuracy = text_BernoulliNB(train_feature_list, test_feature_list, train_class_list, test_class_list)
        test_accuracy_list[1].append(test_accuracy)
        test_accuracy = text_ComplementNB(train_feature_list, test_feature_list, train_class_list, test_class_list)
        test_accuracy_list[2].append(test_accuracy)
        test_accuracy = text_CategoricalNB(train_feature_list, test_feature_list, train_class_list, test_class_list)
        test_accuracy_list[3].append(test_accuracy)
        test_accuracy = text_GaussianNB(train_feature_list, test_feature_list, train_class_list, test_class_list)
        test_accuracy_list[4].append(test_accuracy)

    plt.figure(1)
    plt.plot(deleteNs, test_accuracy_list[0])
    plt.title('Relationship of deleteNs and MultinomialNB_test_accuracy')
    plt.xlabel('deleteNs')
    plt.ylabel('test_accuracy')

    plt.figure(2)
    plt.plot(deleteNs, test_accuracy_list[1])
    plt.title('Relationship of deleteNs and BernoulliNB_test_accuracy')
    plt.xlabel('deleteNs')
    plt.ylabel('test_accuracy')

    plt.figure(3)
    plt.plot(deleteNs, test_accuracy_list[2])
    plt.title('Relationship of deleteNs and ComplementNB_test_accuracy')
    plt.xlabel('deleteNs')
    plt.ylabel('test_accuracy')

    plt.figure(4)
    plt.plot(deleteNs, test_accuracy_list[3])
    plt.title('Relationship of deleteNs and CategoricalNB_test_accuracy')
    plt.xlabel('deleteNs')
    plt.ylabel('test_accuracy')

    plt.figure(5)
    plt.plot(deleteNs, test_accuracy_list[4])
    plt.title('Relationship of deleteNs and GaussianNB_test_accuracy')
    plt.xlabel('deleteNs')
    plt.ylabel('test_accuracy')
    plt.show()