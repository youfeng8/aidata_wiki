# 自然语言处理

标签（空格分隔）： 深度学习

---

[toc]

参考：
[NLP系列之文本分类](https://msd.misuland.com/pd/2884249965817761572?page=1)
[kaggle之电影文本情感分类](https://blog.csdn.net/lijingpengchina/article/details/52250765)

## 基础
[LSTM原理和实现](https://www.zybuluo.com/mdeditor#1111865)
[RNN原理和实现](https://www.zybuluo.com/mdeditor#1111856)


 1. 预处理，包含文本提取，分词，去停用词，去掉低频词（word2vec也有）等等，
 2. 对处理好词数组进行训练词转向量， 如word2vec将词转向量。
 3. 对所有出现到词去重，生成每一个词和向量的index。注意处理前需要增加`PAD`和`UNK` 两个词，`PAD`表示的当语句的词长度训练指定不够时需要在前或者在后补，词向量取`np.zeros(self._embeddingSize)`。`UNK`表示分词后不在词典内的词，词向量取`np.random.randn(self._embeddingSize)`。
 4. 将标签和句子数值化，句子借助第三步生产的词的index。
 5. 初始化训练集和测试集，对数据集进行分割。
 6. 构建模型。
 7. 进行模型训练。



## 一、预处理
### 1.1 文本处理
#### 1.1.1 全角转半角
[python实现全角半角的相互转换](https://www.cnblogs.com/kaituorensheng/p/3554571.html)
```
def strQ2B(ustring):
    """全角转半角"""
    rstring = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 12288: 
            #全角空格直接转换
            inside_code = 32
        elif (inside_code >= 65281 and inside_code <= 65374): 
            #全角字符（除空格）根据关系转化
            inside_code -= 65248
        rstring += chr(inside_code)
    return rstring
```
#### 1.1.1 编码处理

### 1.2 信息提取

#### html提取 
BeautifulSoup
#### 正则

### 1.3 分词
jieba
### 1.4 去停用词
NLTK
停用词表

### 1.5 词频统计

```
from collections import Counter
subWords = ["车辆","车","车主","发动机","维修"] 
wordCount = Counter(subWords)  # 统计词频
sortWordCount = sorted(wordCount.items(), key=lambda x: x[1], reverse=True)

# 去除低频词
words = [item[0] for item in sortWordCount if item[1] >= 5]
```
## 二、词处理
### 2.1 TF-IDF

### 2.2 word2vec

参考：[自然语言处理之word2vec](https://www.cnblogs.com/jiangxinyang/p/9332769.html)
[词向量详解：从word2vec、glove、ELMo到BERT](https://cpu.baidu.com/api/pc/1022/1329713/detail/36786280402577025/news?cpid=mgxYBQrmp8BJ5gxT4iBoH5Jwe6H_kKt0qLPrvmRdJIMUyRhmFGIOwlcK_B_UvAD7HvknDGWq24MD_OgKZMgWyrtAvzdAPFzQaJcpoIo7O4uBI4GmgV8lrwIS4623uTGZ7nM0X6iqDDT0MIpWdLfCcE32vsUoJ3Zaa70Emvfxx2UhQv_Dzd7191jyEWW6afFpYEGYCd-tNTN_iP6Nd3pMdEtyUrXaI7m828kGADhrS0g_9lKpJ6ApwbfHUg5Tlv57lh1klUEnI2bu2CeMh4Cwny-PHosMlxRedZZHUs_a0W1zQodTTUKZIWT7n9kE-fpYmalGBuMPeqy8euA-ECB6yA&scene=0&no_list=1&forward=api&api_version=2&cds_session_id=b475a40e278146e6988744e4ef34963d&cpu_union_id=IDFA_f1ebbd0c92f2dcb9d122ba8457545c3a&rts=8)


word2vec的中心思想:用一个词附近的其他词来表示该词。

word2vec模型其实就是简单化的神经网络，主要包含两种词训练模型：CBOW模型和Skip-gram模型。

CBOW模型根据中心词$W(t)$周围的词来预测中心词；Skip-gram模型则根据中心词$W(t)$ 来预测周围的词。

1）CBOW模型的第一层是输入层，输入的值是周围每个词的one-hot编码形式，隐藏层只是对输出值做了权值加法，没有激活函数进行非线性的转换，输出值的维度和输入值的维度是一致的。

2）Skip-gram模型的第一层是输入层，输入值是中心词的one-hot编码形式，隐藏层只是做线性转换，输出的是输出值的softmax转换后的概率。

**词向量的两个优点：**
1）降低输入的维度。词向量的维度一般取100-200，对于大样本时的one-hot向量甚至可能达到10000以上。
2）增加语义信息。两个语义相近的单词的词向量也是很相似的。

#### 2.2.1 CBOW模型
参考:
[Word2Vec之Skip-Gram与CBOW模型原理](https://www.jianshu.com/p/a2e6a487b385)
[基于CBOW训练模型的word2vec](https://blog.csdn.net/weixin_41624658/article/details/82802254)
[探秘Word2Vec(四)-CBOW模型](https://www.jianshu.com/p/d534570272a6)

CBOW根据上下文预测目标单词，最后使用模型的部分参数作为词向量。
训练方式有：基于Huffman树分层Softmax和负采样。 

它包括三层，分别为输入层，投影层和输出层，

1. 输入层是：
包含context(w)中上下文的２×win（窗口）个词向量。即对应目标单词w，选取其上下文各win个单词的词向量作为输入。再所有的向量的求和

2. 第二层为投影层：
将输入层的２×win个向量做累加求和。

3. 输出层: 
对应一颗二叉树，叶子节点共Ｎ个，对应词典里的每个词(全量的数据构造)。我们是通过哈弗曼树来求得某个词的条件概率的。
假设某个词ｗ，从根节点出发到ｗ这个叶子节点，中间会经过４词分支，每一次分支都可以视为一次二分类。从二分类来说，word2ecv定义分到左边为负类（编码为１），分到右边为正类（编码label为０）。在逻辑回归中，一个节点被分为正类的概率为ｐ，分为负类的概率为１－ｐ。将每次分类的结果累乘则得到$p(w∣Context(w))$。概率$p$在逻辑回归二分类问题中，对于任意样本$x=(x1,x2,x3,...,xn)^T$，利用sigmoid函数，求得分为正类的概率为$hθ(w)=σ(θ^Tx)$，负类概率为$1−hθ(w)=σ(θ^Tx)$。

词向量是预测的附带产物：

1. Huffman树

2. Hierarchical Softmax
分层Softmax
采用了Huffman树构造的条件概率。
概率函数：
![H1.webp-8.7kB][1]
极大似然函数：
![F2.webp-7.1kB][2]
梯度计算求导，计算出$\theta$, $\omega$

3. 负采样
已知词w的上下文Context(w)，需要预测w，
生成它的负采样集合NEG(w)。
对于长度为1的线段，词典D中的每一个词根据词频对应线段的一个长度，具体生成词频区间，对词频取了0.75次幂，这个幂实际上是一种“平滑”策略，能够让低频词多一些出场机会，高频词贡献一些出场机会，劫富济贫。
然后再随机一个数能不能落在词w分配区间上。计算时是通过查表方式，将上述线段M个“刻度”，刻度之间的间隔是相等的，即1/M。一直按刻度往前进，看该刻度是不是点是不分配的词向量的点。

损失函数计算上下文与目标单词之间的点积，采集每一个正样本的同时采集k个负样本。公式的第一项最小化正样本的损失，第二项最大化负样本的损失。现在如果将负样本作为第一项的变量输入，则损失函数结果应该很大。

概率函数：
![F3.webp-9.2kB][3]
极大似然函数：
![F4.webp-6.8kB][4]

梯度计算求导，计算出$\theta$, $\omega$

**训练过程：**

 1. 准备好语料，将训练数据保存为txt文件中。另取一些数据作为测试数据。
 2. 设置一个类class，保存词以及它的哈夫曼树路径、哈弗曼编码、词频
 3. 初始化各类参数，扫描语料库，统计词频，并依据每个词的词频生成生成哈弗曼树。生成哈弗曼树后生成每个词的哈弗曼编码以及路径。初始化输入层词向量syn0以及哈弗曼树上非叶子结点的向量syn1。
 4. 训练，迭代优化。训练过程中就是通过不断的输入，用随机梯度上升的方法，去更新词向量的值(syn0)和非叶子结点处向量的值(syn1)。实质上就是让词向量在词向量空间中找到正确的位置。
训练伪代码如图：

#### 2.2.2 gensim使用
**word2vec API讲解:**
在gensim中，word2vec相关的API都在包gensim.models.word2vec中。和算法有关的参数都在类gensim.models.word2vec.Word2Vec中。算法需要注意的参数有：

 - sentences：我们要分析的语料，可以是一个列表，或者从文件中遍历读出（word2vec.LineSentence(filename) ）。
 - size：词向量的维度，默认值是100。这个维度的取值一般与我们的语料的大小相关，如果是不大的语料，比如小于100M的文本语料，则使用默认值一般就可以了。如果是超大的语料，建议增大维度。
 - window：即词向量上下文最大距离，window越大，则和某一词较远的词也会产生上下文关系。默认值为5，在实际使用中，可以根据实际的需求来动态调整这个window的大小。如果是小语料则这个值可以设的更小。对于一般的语料这个值推荐在[5；10]之间。
 - sg：即我们的word2vec两个模型的选择了。如果是0， 则是CBOW模型；是1则是Skip-Gram模型；默认是0即CBOW模型。
 - hs：即我们的word2vec两个解法的选择了。如果是0，则是Negative Sampling；是1的话并且负采样个数negative大于0， 则是Hierarchical Softmax。默认是0即Negative Sampling。
 - negative：即使用Negative Sampling时负采样的个数，默认是5。推荐在[3,10]之间。这个参数在我们的算法原理篇中标记为neg。
 - cbow_mean：仅用于CBOW在做投影的时候，为0，则算法中的xw为上下文的词向量之和，为1则为上下文的词向量的平均值。在我们的原理篇中，是按照词向量的平均值来描述的。个人比较喜欢用平均值来表示xw,默认值也是1,不推荐修改默认值。
 - min_count：需要计算词向量的最小词频。这个值可以去掉一些很生僻的低  - iter：随机梯度下降法中迭代的最大次数，默认是5。对于大语料，可以增大这个值。
 - alpha：在随机梯度下降法中迭代的初始步长。算法原理篇中标记为η，默认是0.025。
 - min_alpha: 由于算法支持在迭代的过程中逐渐减小步长，min_alpha给出了最小的迭代步长值。随机梯度下降中每轮的迭代步长可以由iter，alpha， min_alpha一起得出。这部分由于不是word2vec算法的核心内容，因此在原理篇我们没有提到。对于大语料，需要对alpha, min_alpha,iter一起调参，来选择合适的三个值。

word2vec是可以进行增量式训练的，因此可以实现一：在输入输入值时可以将数据用生成器的形式导入到模型中；二：可以将数据一个磁盘中读取出来，然后训练完保存模型；之后加载模型再从其他的磁盘上读取数据进行模型的训练。初始化模型的相似度之后，模型就无法再进行增量式训练了，相当于锁定模型了。

```
import time
from gensim.models import Word2Vec
# 模型参数
＃词向量维数
num_features = 300  
＃最小字数
min_word_count = 40 
＃并行运行的线程数
num_workers = 4 
＃上下文窗口大小
context = 10                                                        # 常用词的下采样设置                      
downsampling = 1e-3 

sentences = [["a", "b"], ["c", "d   "]]

model = Word2Vec(sentences, workers=num_workers, \
            size=num_features, min_count = min_word_count, \
            window = context, sample = downsampling)
```
**最常见的应用：**

1. 找出某一个词向量最相近的集合
方法：`model.wv.similar_by_word()`
从这里可以衍生出去寻找相似的句子，比如“北京下雨了”，可以先进行分词为{“北京”，“下雨了”}，然后找出每个词的前5或者前10个相似的词，
比如”北京“的前五个相似词是:{“上海”， “天津"，”重庆“，”深圳“，”广州“}
"下雨了"的前五个相似词是:{”下雪了“，”刮风了“，”天晴了“，”阴天了“，”来台风了“}
然后将这两个集合随意组合，可以得到25组不同的组合，然后找到这25组中发生概率最大的句子输出。
2. 查看两个词向量的相近程度
方法：`model.wv.similarity()`
比如查看"北京"和”上海“之间的相似度
3. 找出一组集合中不同的类别
方法：`model.wv.doesnt_match()`
比如找出集合{“上海”，“天津"，”重庆“，”深圳“，”北京“}中不同的类别，可能会输出”深圳“，当然也可能输出其他的
```
import gensim
import re
import logging
import time
from gensim.models import word2vec

from src.common.utils import dir_and_file_utils

# 数据目录
data_path = "/home/kubernetes/code/nlp/sogou/data"

word_embdiing_path = data_path + "/word_embdiing.txt"

print(word_embdiing_path)
sentences = word2vec.LineSentence(word_embdiing_path)
print("数据行数 = ", list(sentences))

# 进行word2vec模型
model = gensim.models.Word2Vec(sentences, size=200, sg=1, iter=8, min_count=10, workers=6)  

# 持久化模型
word2Vec_path = "./word2Vec.bin"
model.wv.save_word2vec_format(word2Vec_path, binary=True) 

# 加载word2Vec 模型 
model = gensim.models.KeyedVectors.load_word2vec_format(word2Vec_path, binary=True)

# 直接取出词向量 
vector = model.wv[u'借贷']
print("词向量 : ", vector)

# 计算两个词向量的相似度
sim = model.similarity(u'借贷',u'借钱')
print("相似度 : ", sim)

# 获得相近词
similar_words = model.similar_by_word(u'采访',topn =5)
print("相近词 : ", similar_words)

# 获得相关词
most_similar_words = model.most_similar(u'汽车',topn =5)
print("相关词 : ", most_similar_words)
```

### 2.3 句子和标签数值化

如：句子`[["汽车", "销售", "冠军"],["发动机", "维修"], ,["汽车", "维修", "免费"]]` 转换为`[[2,3,5],[4,6,0],[2,6,1]]`
标签`["买","修","修"]`转换为`[0,1,1]`

这里的作用在tensroflow训练传人的数据必须是数值型，所以需要将句子的每一个单词转换为index。还有就是因为在Tensroflow训练时可以直接通过index获取转换为词向量。推理时可以转换为词向量直接推理

**生成词的索引**
```
words = ["汽车", "销售","发动机", "冠军","维修"]

# 获取词向量
wordVec = gensim.models.KeyedVectors.load_word2vec_format("../word2vec/word2Vec.bin", binary=True)

# 初始化数组
vocab = []
wordEmbedding = []

# 添加 "pad" 和 "UNK", 
vocab.append("PAD")
vocab.append("UNK")
wordEmbedding.append(np.zeros(self._embeddingSize))
wordEmbedding.append(np.random.randn(self._embeddingSize))

# 获取词向量并生成数组
for word in words:
    try:
        vector = wordVec.wv[word]
        vocab.append(word)
        wordEmbedding.append(vector)
    except:
        print(word + "不存在于词向量中")

# 词转index
word2idx = dict(zip(vocab, list(range(len(vocab)))))

# 将词汇-索引映射表保存为json数据，之后做inference时直接加载来处理数据
with open("../data/wordJson/word2idx.json", "w", encoding="utf-8") as f:
    json.dump(word2idx, f)
    
# 后续很多地方会用到，如词index转向量
wordEmbedding = np.array(wordEmbedding)
```
**生成去重的标签的索引**
```
labels = ["a", "b", "b"]
uniqueLabel = list(set(labels))
label2idx = dict(zip(uniqueLabel, list(range(len(uniqueLabel)))))

**将去重的词转索引**       
with open("../data/wordJson/label2idx.json", "w", encoding="utf-8") as f:
    json.dump(label2idx, f)
```


**将标签和句子数值化**
```
def _labelToIndex(self, labels, label2idx):
    """
    将标签转换成索引表示
    """
    labelIds = [label2idx[label] for label in labels]
    return labelIds
    
def _wordToIndex(self, reviews, word2idx):
    """
    将词转换成索引
    """
    reviewIds = [[word2idx.get(item, word2idx["UNK"]) for item in review] for review in reviews]
    return reviewIds
```
### 2.4 数值化的句子转向量
```

# 词向量数组，与词的index一致
# "PAD" 对应的词向量 np.zeros(self._embeddingSize)
# ""UNK" 对应的词向量 np.random.randn(self._embeddingSize)
wordEmbedding = [......]
        
# 词嵌入层
with tf.name_scope("embedding"):
    # 利用预训练的词向量初始化词嵌入矩阵
    self.W = tf.Variable(tf.cast(wordEmbedding, dtype=tf.float32, name="word2vec") ,name="W")
    # 利用词嵌入矩阵将输入的数据中的词转换成词向量，维度[batch_size, sequence_length, embedding_size]
    self.embeddedWords = tf.nn.embedding_lookup(self.W, self.inputX)
    # 卷积的输入是思维[batch_size, width, height, channel]，因此需要增加维度，用tf.expand_dims来增大维度
    self.embeddedWordsExpanded = tf.expand_dims(self.embeddedWords, -1)
```

## 三、模型训练

### 3.1 数据拆分
NLP任务中, 被pad和unk的向量应该赋值为zero还是random呢？
比如语义匹配任务，分词后不在词典内的词经常被标为<unk>，处理为相同长度通常会在前或后补<pad>，这两种大家一般选择zero()还是random()来初始化呢？两种方法的区别是什么？
```
def genTrainEvalData(self, x, y, word2idx, rate):
    """
    x: 词的index [[1, 12, 56, 43], [2, 12, 12]]
    y: lable的index [1, 2]
    生成训练集和验证集
    """
    reviews = []
    for review in x:
        # 如果没有达到设定则补齐"PAD"向量
        if len(review) >= self._sequenceLength:
            reviews.append(review[:self._sequenceLength])
        else:
            reviews.append(review + [word2idx["PAD"]] * (self._sequenceLength - len(review)))
   
    # 切割点Index     
    trainIndex = int(len(x) * rate)
    
    # 拆分数据集
    trainReviews = np.asarray(reviews[:trainIndex], dtype="int64")
    trainLabels = np.array(y[:trainIndex], dtype="float32")
    
    evalReviews = np.asarray(reviews[trainIndex:], dtype="int64")
    evalLabels = np.array(y[trainIndex:], dtype="float32")

    return trainReviews, trainLabels, evalReviews, evalLabels
```

### 3.2 传统机器学习
参考：https://blog.csdn.net/u010665216/article/details/78813407

传统机器学习方案是
 
 1. 加载word2Vec模型
 2. for review in reviews:
    1. review 里所有词转向量，np.add(featureVec, model[word])：如review=["天气", "蓝"]转为featureVec=[[0.1,-0.2],[0.3,0.4]]
    2. 所有词向量进行取平均操作， featureVec = np.divide(featureVec, nwords) 如：[[0.1,-0.2],[0.3,0.4]]变成了[[ 0.05, -0.1 ],[ 0.15, 0.2 ]]
    
所有传统机器学习模型获取训练数据方案都是一样的：
```
import logging
import gensim
from gensim.models import word2vec

# 加载word2Vec 模型 
word2Vec_path = "./word2Vec.bin"
model = gensim.models.KeyedVectors.load_word2vec_format(word2Vec_path, binary=True)

# 数据目录
data_path = "/home/kubernetes/code/nlp/sogou/data"

# 分词后数据
cut_words_path = data_path + "/cut_words.csv"

import pandas as pd 
df = pd.read_csv(cut_words_path)
df.head(10)
train_data = [news.split(" ") for news in df["news"].values]
labels = [int(label) for label in df["label"].values]

import numpy as np
def makeFeatureVec(words, model, num_features):
    '''
    对段落中的所有词向量进行取平均操作
    '''
    featureVec = np.zeros((num_features,), dtype="float32")
    nwords = 0.

    # Index2word包含了词表中的所有词，为了检索速度，保存到set中
    index2word_set = set(model.index2word)
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1.
            featureVec = np.add(featureVec, model[word])

    # 取平均
    featureVec = np.divide(featureVec, nwords)
    return featureVec

def getAvgFeatureVecs(reviews, model, num_features):
    '''
    给定一个文本列表，每个文本由一个词列表组成，返回每个文本的词向量平均值
    '''
    counter = 0

    reviewFeatureVecs = np.zeros((len(reviews), num_features), dtype="float32")

    for review in reviews:
        if counter % 5000. == 0:
            print("Review %d of %d" % (counter, len(reviews)))

        reviewFeatureVecs[counter, ] = makeFeatureVec(review, model, num_features)

        counter = counter + 1
    return reviewFeatureVecs

num_features = 200
%time trainDataVecs = getAvgFeatureVecs(train_data, model, num_features)
print(trainDataVecs[0:10])
```

#### 3.2.1 LR
```
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import numpy as np

log_reg = LogisticRegression()

print("逻辑回归分类器10折交叉验证得分: ", np.mean(cross_val_score(log_reg, trainDataVecs, labels, cv=10, scoring='accuracy')))
```
#### 3.2.2 SVM
```
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import numpy as np

# 注意核函数的选择
svc=SVC(kernel='poly',degree=2,gamma=1,coef0=0)
print("SVM分类器10折交叉验证得分: ",
      np.mean(cross_val_score(svc, trainDataVecs, labels, cv=4, scoring='accuracy')))
```
#### 3.2.3 随机森林
```
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

forest = RandomForestClassifier(n_estimators = 100, n_jobs=2)

print("随机森林分类器10折交叉验证得分: ",
      np.mean(cross_val_score(forest, trainDataVecs, labels, cv=4, scoring='accuracy')))
```
#### 3.2.4 贝叶斯
```
from sklearn.naive_bayes import GaussianNB as GNB
from sklearn.model_selection import cross_val_score
import numpy as np

model_GNB = GNB()
# model_GNB.fit(trainDataVecs, labels)

print("高斯贝叶斯分类器10折交叉验证得分: ", np.mean(cross_val_score(model_GNB, trainDataVecs, labels, cv=10, scoring='accuracy')))
```

### 3.3 深度学习
#### 3.3.1 textCNN
参考：[TextCNN模型原理和实现](https://www.cnblogs.com/bymo/p/9675654.html)

优点：能够更好地捕捉局部相关性，类似n-gram

将卷积神经网络CNN应用到文本分类任务，利用多个不同size的kernel来提取句子中的关键信息（类似于多窗口大小的ngram），从而能够更好地捕捉局部相关性。如词向量长为128，则卷积核类型为[2,128]，[3,128]，[4,128]，每种类型卷积核有两个，卷积是即各个位置的元素相乘再相加。

![F2.png-71.1kB][5]

1. Embedding：第一层是图中最左边的7乘5的句子矩阵，每行是词向量，维度=5，这个可以类比为图像中的原始像素点。
2. Convolution：然后经过 kernel_sizes=(2,3,4) 的一维卷积层，每个kernel_size 有两个输出 channel（两个同样大小的卷积核：`Conv1D(filters=2, kernel_size=kernel_size, strides=1)）`。
3. MaxPolling：第三层是一个1-MaxPooling层，这样不同长度句子经过pooling层之后都能变成定长的表示。如："我觉得这个地方景色还不错，但是人也实在太多了"，虽然前半部分体现情感是正向的，全局文本表达的是偏负面的情感，利用1-max pooling能够很好捕捉这类信息。
4. FullConnectionAndSoftmax：最后接一层全连接的softmax层，输出每个类别的概率。

**TextCNN模型结构图:**

![F1.png-63.8kB][6]

#### 3.3.2 charCNN

#### 3.3.3 Bi-LSTM
参考：
[LSTM](https://www.cnblogs.com/jiangxinyang/p/9362922.html)
[BiLSTM](https://www.jianshu.com/p/4999861c26a7)
[BiLSTM介绍及代码实现](https://www.jiqizhixin.com/articles/2018-10-24-13)

前向的LSTM与后向的LSTM结合成BiLSTM。

优点： LSTM更好的捕捉到较长距离的依赖关系，LSTM对句子进行建模还存在一个问题：无法编码从后到前的信息（一些奇怪的句子），通过BiLSTM可以更好的捕捉双向的语义依赖。

BiLSTM输出分为两种：横向拼接

1. $o^{(t)}=V(h_L^{(t)}+h_R^{(t)})+c$，包含了前向与后向的所有信息，如情感分类任务
2. $o^{(t)}=V(h_L^{(t)}+h_R^{(n-t)})+c$，



#### 3.3.4 Bi-LSTM + Attention

优点：Attention是先计算每个时序输出的权重，然后将所有时序的输出进行加权和作为特征向量，然后进行softmax分类。在实验中，加上Attention确实对结果有所提升。

Attention 机制：

基本思想就是，打破了传统编码器-解码器结构在编解码时都依赖于内部一个固定长度向量的限制。Attention的实现是通过保留LSTM编码器对输入**蓄力**的中间输出结果，然后训练一个模型来对这些输入进行选择性的学习并且在模型输出时将输出序列与之进行关联。

参考:[易于理解的一些时序相关的操作(LSTM)和注意力机制(Attention Model)](https://blog.csdn.net/wangyanbeilin/article/details/81350683)
[LSTM/RNN中的Attention机制](https://www.jianshu.com/p/4b49a1964ddc)
[自然语言处理中的Attention机制总结](https://blog.csdn.net/hahajinbu/article/details/81940355)

#### 3.3.5 RCNN
优点：textCNN的基础上使用Bi-LSTM对词向量进行扩充。

流程：

1. 利用Bi-LSTM获得上下文的信息，类似于语言模型。
2. 将Bi-LSTM获得的隐层输出和词向量拼接[fwOutput, wordEmbedding, bwOutput]。
3. 将拼接后的向量非线性映射到低维。(降维)
4. 向量中的每一个位置的值都取所有时序上的最大值，得到最终的**特征向量**，该过程类似于max-pool。
5. softmax分类。

举例：　
 
 1. 单词A的向量 = embeddedWords = [0.1, 0.3, 0.3]
 2. Bi-LSTM作用下，前向list_m输出list_fw_h=[0.6,0.9]，后向list_m输出list_bw_h=[0.5,0.6], 
 3. 单词A的新的向量 = [list_fw_h, embeddedWords, list_bw_h] = [0.6, 0.9, 0.1, 0.3, 0.3, 0.5, 0.6]
 4. 在对文件进行降维特征分解，选取特征向量较大的值。


#### 3.3.6 Adversarial LSTM
Adversarial LSTM的核心思想是通过对wordEmbedding上添加噪音生成对抗样本，将对抗样本以和原始样本同样的形式喂给模型，得到一个AdversarialLoss，通过和原始样本的loss相加得到新的损失，通过优化该新的损失来训练模型，作者认为这种方法能对word embedding加上正则化，避免过拟合。

#### 3.3.7 Transformer
Transformer：不用进行词训练

在Encoder中

1. Input 经过嵌入后，要做位置编码，每个输入单词通过词嵌入算法转换为词向量。
2. 然后是自注意力层，
3. 再经过前馈神经网络层，
4. 每个子层之间有残差连接。

在Decoder中

1. 也有 positional encodings，Multi-head attention 和 FFN，子层之间也要做残差连接，
2. 但比 encoder 多了一个 Masked Multi-head attention，
3. 最后要经过 Linear 和 softmax 输出概率。

每个解码器都可以分解成两个子层。

 - 自注意力层，从编码器输入的句子首先会经过一个自注意力（self-attention）层，这层帮助编码器在对每个单词编码时关注输入句子的其他单词。如编码“it”这个单词的时，注意力机制的部分会去关注“The Animal”，将它的表示的一部分编入“it”的编码中。
 - 前馈神经网络层，自注意力层的输出会传递到前馈（feed-forward）神经网络中。每个位置的单词对应的前馈神经网络都完全一样（译注：另一种解读就是一层窗口为一个单词的一维卷积神经网络）。

**注意：**解码器中也有编码器的自注意力（self-attention）层和前馈（feed-forward）层。除此之外，这两个层之间还有一个注意力层，用来关注输入句子的相关部分（和seq2seq模型的注意力作用相似）。

自注意力层：

1. 为编码器的每个输入单词创建三个向量，
即查询向量Q、键向量K和值向量V，这些向量在维度上比词嵌入向量更低，如维度是64。
2. 计算出Q、K和V。如3个单词[A,B,C]
    1. $Q_A = Q * X_A,K_A = K * X_A,V_A = V * X_A, $
    1. 对单词A打分，相当于计算相似性，$[Q_A*K_A, Q_A*K_B,Q_A*K_C]$
    2. 将计算出的相似性用softmax归一化处理，权重系数。
    3. 计算Value值的加权和，利用计算出来的权重系数对各个Value的值加权求和，就得到了我们的Attention的结果。
3. 残差
$H(x)=F(x)+ x$z
4. Add Norm
在这里LayerNorm中每个样本都有不同的均值和方差，不像BatchNormalization是整个batch共享均值和方差。
多头注意力得到的输入[1,512]和输入x_1[1,512]直接相加[1,512]。再对这一层[x_1,x_2,...,x_n]进行层归一化。

多头：
多个“表示子空间”（representation subspaces）
怎么理解？，
我们将其与卷积联系在一起，如果一组Q，K，V相当于一个卷积核，认为不同的卷积核会捕获不同的局部信息，能够提取一个特征[1,64]，那么8组Q，K，V相当于8个卷积核，能够提取八组特征。这个极大的扩展了模型的复杂度，能够学习到更多文本特征。

前馈网络层：

就是全连接，这边论文中所述是两层。
第一层[1,512] * [512,512*4] = [1,2048]
Relu
第二层[1,2048] * [2048,512] = [1,512]

[图解什么是 Transformer](https://www.jianshu.com/p/e7d8caa13b21)
[BERT大火却不懂Transformer？读这一篇就够了](https://baijiahao.baidu.com/s?id=1622064575970777188&wfr=spider&for=pc)

### 3.4 预训练模型
#### 3.4.1 ELMo
#### 3.4.2 BERT

## 四、模型推理


## 五、优化方案


  [1]: http://static.zybuluo.com/tc1052400205/sy8rwp8k127v7twfhbufpwp5/H1.webp
  [2]: http://static.zybuluo.com/tc1052400205/8a7lx1ctpoukxp8n523y4itv/F2.webp
  [3]: http://static.zybuluo.com/tc1052400205/dhws20q69him68hvcdc9qp5j/F3.webp
  [4]: http://static.zybuluo.com/tc1052400205/dq4yvhwrlh2aytwh1s6cvw75/F4.webp
  [5]: http://static.zybuluo.com/tc1052400205/ah7bsaojpay5tid2a7oq7yum/F2.png
  [6]: http://static.zybuluo.com/tc1052400205/6xdw3fddcffc8tsvsfxi1aw1/F1.png