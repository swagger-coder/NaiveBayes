import random

import numpy as np

'''数据集 六个样本'''
def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]
    return postingList, classVec

''' 获取词汇表'''
def createVocabList(dataSet):
    vocabSet = set()
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

'''将词条向量化，向量维度等于词汇表长度'''
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec

'''训练算法：求条件概率矩阵'''
def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)  # 训练样本数量
    numWords = len(trainMatrix[0])  # 词汇表长度
    pAbusive = sum(trainCategory) / float(numTrainDocs)  # p(c1) 是侮辱性留言的概率
    #     p0Num = numpy.zeros(numWords); p1Num = numpy.zeros(numWords)
    #     p0Denom = 0.0; p1Denom = 0.0  # 分母值 各自类别的词条数
    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)  # 将所有词出现的次数初始化为1
    p0Denom = 2.0
    p1Denom = 2.0  # 分母值 各自类别的词条数
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]  # 统计词频
            p1Denom += sum(trainMatrix[i])  # 总数累加
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    # p(wi|ci) 使用log是为了防止下溢出
    p1Vect = np.log(p1Num / p1Denom)
    p0Vect = np.log(p0Num / p0Denom)
    return p0Vect, p1Vect, pAbusive


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    # 因为转成log 所以都是加；log(pClass1)是p（ci）
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)
    p2 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
    if p1 > p2:
        return 1
    else:
        return 0


def testingNB():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    # 生成词向量矩阵
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB0(np.array(trainMat), np.array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))


def bagOfWord2VecMN(vocabList, inputSet):
    '''词袋模型'''
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

def  textParse(bigString):
    import re
    listOfTokens = re.split('\W',bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def spamTest():
    docList = []; classList = []; fullText = []
    for i in range(1,26):
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    trainingSet = [i for i in range(50)]
    testSet = []

    # 分配十个作为测试集
    for i in range(10):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])

    # 训练
    trainMat = []; trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])

    p0V, p1V, pSpam = trainNB0(np.array(trainMat), np.array(trainClasses))

    # 错误数
    errorCount = 0

    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])
        if classifyNB(np.array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            print('error docIndex', docIndex)
            errorCount += 1

    print('the error rate is: ', float(errorCount)/len(testSet))

if __name__ == '__main__' :
    print('侮辱性文本分类')
    testingNB()
    print()
    print('垃圾邮件分类')
    spamTest()






