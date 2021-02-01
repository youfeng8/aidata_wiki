#【learning from imbalanced data sets】第二章、不平衡分类的基础

标签（空格分隔）： AI_不平衡学习

---

## 阅读笔记和思考

样本不平衡问题难点定义：

 - 小样本不具备代表性，实际较为随机，没有高区分度特征。
 - 类别重叠，没有明显的边界。

## 摘要

类不平衡是许多现实世界中的分类数据集，它涉及问题中不同类的示例数量的不成比例。众所周知，此问题由于其面向准确性的设计而阻碍了分类器的性能，这通常会使少数类被忽略。在本章中，介绍了有关类不平衡问题的基础。第2.1节对不平衡分类进行了正式描述，并说明了为什么需要使用特定方法来解决此问题。 2.2节专门介绍存在不平衡分类的不同应用领域。最后，第2.3节介绍了一些关于不平衡分类的案例研究，包括几个测试台，可以比较设计用于解决不平衡分类问题的算法。本书的其余部分将考虑其中一些案例研究，以分析所讨论的不同方法的行为。

## 2.1 形式描述

类分布不均的任何数据集在技术上都是不平衡的。但是，当问题的每个类别的示例数之间存在显着（甚至在某些情况下为极端）不相称时，则认为数据集不平衡。换句话说，当代表一个类别的示例数量远低于其他类别的示例数量时，就会发生类别不平衡。因此，一个或多个类在数据集中可能不足。由于收集的原始数据满足此定义的实际应用程序数量众多，因此这种简单的定义引起了研究人员和实践者的广泛关注。例如，已知受此问题困扰的应用程序包括：故障诊断[84，88]，异常检测[37，74]，医学诊断[54]，电子邮件折叠[10]，面部识别[49]或漏油的检测[42]等。

大多数不平衡的分类文献都致力于二元分类问题，其中一类明显多于另一类（因此代表性不足）。然而，偏态分布也存在多类问题[25，79]。第8章解决了不平衡的多类分类问题，尽管在后续章节中针对二进制分类的许多概念也适用于多类问题。

在两类问题中，少数派（代表性不足）通常被称为积极派，而多数派被视为否定派。这些术语在文献中可以互换使用。
![image_1etc2u4841rrpksd1djnjkceir9.png-368.2kB][1]
**图2.1 比率为1：100的两类不平衡问题的示例**

在下面，我们将用一个简单的合成数据集说明不平衡分类的问题，如图2.1所示。这是比率为1：100的两类问题，也就是说，对于每个正面/少数派类别的示例，都有100个负面/多数派类别的示例。正面示例带有蓝色星号“ *”，而负面示例则带有红点“。”。可以清楚地看到，肯定的类别的代表人数不足，并且很难定义一个将两个类别分开的决策边界。

与1：100表示法不同，表示两类问题的不平衡程度的另一种常见方法是不平衡率（IR）[57]。 IR定义为否定类别示例的数量除以肯定类别示例的数量，可用于根据其IR对不同的数据集进行排序。因此，在我们的示例中，IR为100。无论如何，正如我们将在后面解释的那样，必须考虑到IR并不总是能够很好地估计数据集的难度。

不平衡问题中的主要问题之一是，从应用的角度来看，代表性不足的类通常是问题的关注类[12]。例如，根据图2.1，可能会认为我们正在处理一种医学应用，我们应该使用活检后测量的两种不同特征来区分特定类型癌症的良性和恶性肿瘤。在这种情况下，正确识别恶性肿瘤比良性肿瘤更为重要，因为未检测到的恶性肿瘤的后果可能是致命的，而当肿瘤为良性时假阳性就不会那么有害。显然，两个分类都希望100％的准确性。但是，事实是，分类器通常远非完美，它们在多数类别中往往具有很高的准确性，而在少数类别中却获得较差的结果（接近0％）。

![image_1etc2un5hkr61osp1vos1ioj1eipm.png-342.6kB][2]
**图2.2 从图2.1中的数据中学到的两个模型的示例。 测试数据是从相同的分布和相同的IR获得的。 描述了每个模型的决策边界**

标准分类器学习算法通常偏向多数类，因为正确地预测那些实例的规则被加权为正，从而有利于准确性度量或相应的成本函数。否则，可以忽略预测少数类中的示例的特定规则（将它们视为噪声），因为首选更通用的规则。结果，少数群体实例比多数群体实例被错误分类。为了说明这个问题，图2.2给出了由图2.1中的问题的两个模型获得的决策边界。请注意，我们已经使用相同的数据分布和IR为问题生成了一个测试集，以便正确估计每个模型的准确性。在左侧，模型1获悉应将每个示例标记为否定，即，它是微不足道的模型。实际上，这样一个简单的模型在测试数据中达到了99.01％的准确度（正确分类的示例的百分比），而在阳性分类示例中却获得了0％的准确度。即，否定类的所有示例均已正确分类，而肯定类中的所有示例均未正确分类。在右侧，模型2有所不同，它了解到部分特征空间属于正类（因为可以在决策边界处观察到），鉴于正类为，这似乎是更可取的我们感兴趣的类别。但是，从性能上看，在测试集上获得的准确度为98.61％，低于模型1所获得的准确度。尽管在全局准确度方面不那么精确，但是第二个模型能够区分两个类别，其中模型1是无法做到。

从这个例子中可以学到两个教训。

 1. 在不平衡情况下，精度不再是一种适当的度量，因为它无法区分正确分类的不同类别示例的数量。因此，这可能会导致错误的结论，即，如果分类器将所有示例分类为否定的，则在IR值为9的数据集中达到90％的准确性。因此，在这种情况下，需要更多的信息量度以评估模型的质量，例如ROC，几何平均值，f量度，精度或召回率。在第3章中讨论了如何在此框架中衡量模型的质量。
 2. 我们需要以某种方式构造偏向少数群体的分类器，而不会损害多数群体的准确性。因此，这本书将展示如何实现这一点，以找到有意义的不平衡数据模型。例如，如在我们的玩具示例中获得的Model 2。

为了达到正确区分少数群体的目的，已经开发了大量技术。这些技术可以根据其处理问题的方式分为四大类。

 1. 算法级别的方法（也称为内部方法），尝试调整现有的分类器学习算法，使学习偏向少数群体[7、47、50]。为了进行适配，需要相应分类器和应用领域的特殊知识，以便理解为什么当类分布不均匀时分类器失败的原因。有关这些类型的方法的更多详细信息，请参见第6章。
 2. 数据级别（或外部）方法旨在通过重新采样数据空间来重新平衡类分布[9、24、56、70]。这样，避免了学习算法的修改，因为由不平衡引起的影响在预处理步骤中减小了。这些方法将在第5章中进行深入讨论。
 3. 成本敏感型学习框架介于数据和算法级别方法之间。数据级别转换（通过向实例添加成本）和算法级别修改（通过修改学习过程以接受成本）[13、48、86]均被合并。分类器偏向少数群体，原因是假定该类别的误分类成本较高，并试图使这两个类别的总成本误差最小。第四章概述了对类不平衡问题的成本敏感方法。
 4. 基于集成的方法通常由集成学习算法[59]和以上技术之一，特别是数据级别和成本敏感技术[27]的组合组成。新的混合方法在集成学习算法中添加了数据级别的方法，通常在训练每个分类器之前对数据进行预处理，而成本敏感的集成而不是修改基本分类器以便在学习过程中接受成本，通过以下方法指导成本最小化集成学习算法。基于集成的模型在第7章中进行了详细描述。
 
![image_1etc30o8r16co1gvkms7svaah813.png-171.9kB][3]
**图2.3 不平衡数据集中的困难示例。 (a)类别重叠。 (b)小杂项**

我们提出了几种解决类别不平衡问题的方法。但是，我们必须强调，偏斜的数据分布本身不会阻碍学习任务[34，71]。因此，如前所述，IR对于理解不平衡问题的难度并不是真正有用的。真正的问题是通常会出现与此问题相关的一系列困难

 - 样本量小：通常，不平衡的数据集缺少少数类示例。在[36]中，作者报告说，当少数派类别的示例数量具有代表性（固定不均衡比例）时，由不均衡类别分布引起的错误率会降低。这样，尽管类别分布不均匀，但可以更好地学习正面类别示例定义的模式。但是，在处理现实世界中的问题时，这一事实通常是无法解决的。
- 重叠或类可分离性（图2.3a）：指两个类的示例在某种程度上混合在特征空间中的事实，即无法清晰地确定决策边界。当数据集中存在重叠时，判别规则的归纳变得更加困难。结果，提取了更通用的规则，这些规则对少量实例（少数群体实例）进行了错误分类[29]。应该考虑到在类之间没有重叠的情况下，实例的类分布变得不那么重要，因为任何简单的分类器学习算法都将能够独立于IR解决分类问题。
- 小析取物（图2.3b）：当少数类代表的概念由子概念构成时，就会出现此问题[81]。在大多数问题中，小析取词是隐含的，它们的存在通常会增加问题的复杂性，因为其中的实例数量通常不平衡。

在本章中，我们简要介绍了类不平衡框架中出现的问题。由于这些困难的重要性，所有这些问题将在第9章中进行深入讨论。

## 2.2 应用

在本节中，我们回顾存在类不平衡的几个应用程序域。我们将介绍在每个领域中发现的不同用例，并参考已应用特定技术处理类不平衡的论文。我们承认，本节的目的不是要彻底审查所有与类不平衡有关的申请文件，而是要对它们进行概述，以显示此问题在实际应用中的重要性。有关存在类不平衡问题的应用程序的详尽概述，请参见[33]。

除了Kubat等人的著作外，我们主要关注最新出版物。 [40–42]。这些论文是解决类不平衡问题的开创者。他们处理从卫星雷达图像中发现漏油的问题。在卫星图像中，溢油的反射率较小，因此可以在图像中识别出来。但是，问题在于，除了漏油以外，还有其他原因导致反射率降低。例如，在下雨，藻类或风的情况下。这种负面的类别被称为外观相似，使溢油的区分变得困难。此外，其数据集中的相似项（896）比溢油事故（41）的IR超过20。作者描述了在其系统开发中发现的一系列困难并提出了可能的解决方案。首先，他们提出了一种称为单面选择的数据采样方法[40]，它是对Tomek Links方法[76]的改进。之后，应用了KNN。其次，他们开发了一种称为SHRINK的特定算法（内部方法），其中在学习算法中引入了g均值性能度量，以提高类别重叠的不平衡问题的性能[41]。

其余应用程序范围为2012年至2018年，我们将根据其应用领域在不同的部分中对其进行描述。在进行每项工作之前，表2.1总结了所考虑的申请文件。它们按出版年份排序。对于每部作品，除了出版年份外，还介绍了应用领域，其主要目标，用于处理类不平衡的技术种类以及参考资料。

**表2.1存在类不平衡问题的ML和DM应用**
|a|a|
|-|-|
|||

### 2.2.1 工业领域

除了Kubat等人的工作。 [42]旨在检测卫星雷达图像中的溢油，还有其他工程应用需要特定的方法来处理数据不平衡问题。其中，[32，44]解决了半导体中的缺陷检测。通常，故障或缺陷预测是类别不平衡问题的另一个典型示例，因为通常有缺陷的产品比正确生产的产品少得多。但是，在制造业中，重要的是要正确检测哪些产品有缺陷，以避免客户不满意和退货。

在李等人的工作中。 [44]，对半导体中的缺陷检测的两个不同问题进行了研究。第一个是蚀刻工艺数据，而第二个是化学气相沉积工艺数据。本研究考虑了KNN和SVM的几种变体。测试了三种类型的方法来解决偏斜的数据分布：使用随机欠采样，SMOTE和NCR的数据级别；成本敏感的方法和基于集成的模型，例如SMOTEBoost。另外，一类分类[75]被认为是从积极的类分类数据中学习的替代方法。

在[32]的情况下，作者研究了图像的缺陷检测，设计了一个完整的系统来从图像开始检查半导体。这些图像用于使用不同的方法提取多个特征。为了克服类别的不平衡，提出了SMOTE的新修改版本。将此修改引入到集成机制中，在该机制中，通过使用堆栈对分类器进行修剪和融合。

Zhu等人发表了另一个具有挑战性的用例。 [90]，其中考虑了电力系统短期电压稳定性评估。人们应该考虑到，这种预测机制可以帮助避免由于动态用电负载而造成的停电。显然，对电力系统不稳定的错误分类会导致不可逆的电压崩溃或灾难性的停电，而对错误的情况进行标签错误则费用要低得多。也就是说，必须解决阶级失衡的问题。为此，提出了一种基于预测的非线性SMOTE以及成本敏感的决策树算法，以将决策偏向那些稀缺但有价值的不稳定案例。

在[63]中，作者处理了风机故障的预测。这是一项重要的任务，因为风电场的大部分运营成本是由于其维护以及风电场与工业区之间的距离而引起的。能够自动监视，诊断和预测风力涡轮机的状态是降低维护成本的最佳方法。在这项工作中，作者研究了这种情况下的类别不平衡问题，通常情况下，正常运行可获得的数据要多于故障产生的数据。为此，测试了解决类别不平衡的不同模型（数据重采样，成本敏感和集成）。他们的分析集中在不平衡比率对系统的预测能力的影响上，并表明，以成本敏感型方法修改的基于旋转林的集成体是最可靠的方法。在[73]中考虑了相同的工程问题，但是，在这种情况下，作者提出了一种成本敏感的大利润分配机，以减轻类不平衡问题。否则，在[83]中开发了SMOTE和EasyEnsemble之间的杂交。

### 2.2.2 信息技术

我们将信息技术的应用分为四个不同的子类别：软件缺陷预测，网络分析，文本挖掘和计算机视觉。

软件缺陷预测已成为质量保证团队发现可能的故障的必要工具。软件的大小和复杂性正在增加，因此需要有效的检查和测试方法。在这种情况下，该想法包括从源代码中获取不同的指标，以便区分有缺陷的组件和无缺陷的组件。 NASA在其“度量数据程序”中发布了几个数据集，其中许多度量被视为分类问题的特征。由于无缺陷和有缺陷的模块数量差异很大，这些数据集已在我们已审查的应用论文中使用[43，62，72]。

在[72]中，原始的不平衡二元类数据集被转换为几个平衡的多类数据集，随后通过分解策略解决了这些问题[26]（使用了三种不同的分解，随机校正码，一对多和一对所有）。不同的是，罗德里格斯等。 [62]研究了解决软件缺陷预测中的类不平衡问题的不同方法的效果。在数据级别，考虑了随机欠采样和过采样以及SMOTE。还测试了一种成本敏感的方法，并使用基于集成的模型（SMOTEBoost和RUSBoost）完成了该研究。文献[43]的作者并未考虑处理类别不平衡的特定技术，而是声称将平均概率集成模型（每个基本分类器给出的概率进行平均）与FS方案相结合就足够了。在如此困难的情况下。

还有许多用例涉及网络数据分析。在[30]中，作者处理了将P2P僵尸网络网络流量与正常流量区分开的问题。问题是不平衡，因为正常流量示例比异常（僵尸网络）流量示例更多。他们考虑将随机森林与随机欠采样和对成本敏感的学习（引入随机森林）一起使用，作为克服类别失衡的工具。

在[80]中考虑了网络流量分类，其中应将以太网网络上的流量分类为不同的类别。由于大多数网络流量是HTTP或HTTPS（网络流量），因此此问题显然是不平衡的，而对应于不同应用程序的跟踪还有许多其他类型。为了提高这种情况下的性能，作者提出了一种使用Adaboost并在每次迭代中平衡数据集的新集成策略。

[15]中的作者处理了网络分析和信息安全之间的难题。他们设计了一种用于通过网络流量分析检测移动恶意软件的系统。为此，使用了用于类别不平衡的不同技术，例如SMOTE，然后是SVM，对成本敏感的SVM和对成本敏感的C4.5决策树。此外，提出了一种新的内部解决方案，称为基于不平衡数据归类的分类。

在其他方面，ForesTexter [82]是专门为分类文本数据而开发的。在文本分类中，找到具有20–40个类别的数据集并不少见，因此，其中一个被低估的可能性很高。在这种方法中，作者提出了对RandomForest算法的改进，以解决不平衡文本挖掘问题。首先，根据特征对多数和少数族裔的区分能力对不同特征进行了采样。然后，不是使用经典的RandomForest算法学习，而是使用SVM分类器对树的每个节点中的数据进行分区。

计算机视觉在许多应用中还受到类别不平衡的影响。在[87]中，作者处理了从图像识别对象的问题。作者提出的解决方案由转移AdaBoost和改良的SMOTE加权版本组成。正如通常在许多计算机视觉任务中所做的那样，迁移学习被认为需要SMOTE适应此情况。

### 2.2.3 生物信息学

在生物信息学和生物技术中，蛋白质研究是近年来研究人员关注的领域之一。蛋白质结构和功能的鉴定是该研究领域中非常重要的问题，因为它们与生物的功能直接相关。蛋白质分类是解决问题的最有效方法之一，但是蛋白质数据集始终不平衡，因此需要特定的技术。但是，蛋白质鉴定并不是生物信息学中存在失衡的唯一应用。作为示例，我们还将展示在发现类不平衡的情况下细胞识别的应用。

微小RNA（核糖核酸）前体的检测可以从分类的角度看，就像在[45]中所做的那样。微小RNA在与植物和动物基因表达的转录后调控有关的非常复杂的遗传过程中起着至关重要的作用。在[45]中，在根据数据创建分类问题的过程之后，首先使用FS方法对所得结果具有偏斜的类分布的数据集进行了预处理，然后进行了混合预处理（SMOTE +不同次随机欠采样）。为了创建几个平衡的数据集。最后，训练了三个不同的基本分类器（SVM，RF和KNN）以形成12个分类器的整体（因为使用不同的随机欠采样运行创建了4个不同的欠采样数据集）。

杨等。 [85]而不是只关注一个生物信息学问题，而是考虑了几个不平衡的生物信息学问题，以便在不平衡域中测试其新开发的方法。更具体地说，他们考虑了四个不同的用例：miRNA鉴定，蛋白质定位预测，DNA序列启动子鉴定以及蛋白质磷酸化谱预测激酶底物。作者提出的样本子集优化技术解决了所有这些问题。此方法是欠采样模型，致力于从多数类中选择最有用的实例，以对数据集进行智能欠采样。

[78]的作者处理了另一个问题，称为接触图预测，这是蛋白质结构预测的一部分，只有大约2％的阳性样本。给定数据集中的示例数量，作者考虑了基于MapReduce [19]的系统，以克服被认为是大数据问题的问题。在多个MapReduce阶段中，使用随机过采样对数据集进行过极端采样后，执行了进化特征加权方法。这种极端的过度采样包括：由于数据集的特殊性，使得数据集的少数类的实例数比多数类的实例数多。最后，考虑使用RandomForest分类器学习最终模型。

Dai [18]的工作集中在蛋白质数据分类上。由于存在大量的类，因此将多类问题简化为多个二进制分类问题（与类数一样多）。一旦执行了此转换，鉴于应将一个类别与所有其他类别区别开来（其示例总数变得更大），每个新的二元问题的不平衡率都会大大增加。为了解决这个问题，作者考虑了反向随机欠采样的使用，在这种方法中，多数类多次严重欠采样，从而创建了具有来自少数类的更多实例的数据集。之后，每个数据集都将用于学习新的模糊SVM，以形成一个整体。

审查的最后一个生物信息学用例包括一个细胞识别问题[35]，其中应在获得的HEp-2图像中检测到有丝分裂细胞（处于有丝分裂阶段的细胞，即当染色体分为两个核时）。间接免疫荧光测定后。为了检测结缔组织疾病，这种分析很有趣。有丝分裂细胞识别是一个不平衡的问题，因为最常见的细胞状态不是有丝分裂而是间期。例如，这项工作考虑的数据集有70个有丝分裂细胞，与1457个间期细胞形成对比。使用不同的技术（例如随机欠采样，SMOTE和一侧选择）对数据集进行平衡。此外，实验中还考虑了特定的合奏方法。

### 2.2.4 医学

处理医疗问题的应用程序是类别分布不平衡的典型示例。在几个子类别中，可以找到这些类型的问题，例如诊断，预测，监视或质量控制。此后，我们回顾一些例子，其中阶级失衡起着关键作用，以获得成功的决策支持系统。我们将从质量控制工作开始，然后是预测和诊断方面的应用。

#### 2.2.4.1 质量控制

质量控制是指那些通过使用DM技术来改善医疗服务的问题。在[91]中，作者开发了一种旨在预测肺癌患者术后预期寿命的系统。考虑到用于预测的分类周期（1年），在假定间隔内幸存的患者人数显着高于死亡人数。此外，将死亡误分类为生存比在相反方向犯错误要有害得多。这种系统可以帮助临床医生确定应该选择哪个患者进行手术，还可以确定术后风险最大的患者。作者使用改进的SVM能够解决每个示例的成本，从而解决了该问题，该SVM还引入了Boosting机制中，用于解决分类不平衡的问题。作者在从模型中提取可解释的规则方面付出了巨大的努力，以获取白盒模型。

Azari等人的工作。 [4]专注于紧急服务。他们开发了一种系统，用于预测患者在医院急诊室待多久。停留时间长（超过14小时）的患者占床位和费用的大部分，而不到患者的10％。因此，有趣的是要了解谁以及为什么要待很长时间才能改善急诊室和医院的资源管理。显然，作者处理了一个不平衡问题，为此他们设计了一种新的集成方法，其中k-均值聚类与随机或SMOTE过采样结合在一起。

评估急诊服务质量的一种方法是将多发伤患者的服务结果与智能系统获得的预期结果进行比较。为了改善这些类型的智能系统，在[65]中，作者进行了一项针对不同数据级别技术的研究，以处理预测多发性创伤患者生存状态时存在的类别不平衡问题（幸运的是，有存活的患者多于死亡的患者。作者主要致力于通过C4.5决策树开发可解释的模型。但是，他们需要利用从SMOTE到Tomek Links的各种外部方法来处理数据中偏斜的类分布。

#### 2.2.4.2 医疗诊断

到目前为止，医学诊断是应用领域和子类别，在其中可以找到更多不平衡的示例。作为第一个例子，在[11]中，开发了一种计算机辅助系统，用于在计算机断层扫描图像中检测肺结节。肺结节是早期肺癌诊断的重要临床指征。幸运的是，还有更多的图像没有结节，从而造成了不平衡的分类问题。作者开发了一个完整的系统，其中提出了将使用过采样和欠采样的混合概率采样与随机子空间方法相结合来构造分类器集合的方法。

同样，在[1]中作者处理了结节分类，但在这种情况下是针对甲状腺结节。开发了完整的系统，将结节分为恶性或良性。同样，这是分类失衡的经典例子，良性结节患者比恶性结节患者多得多。为了在学习不同分类器（SVM，C4.5，KNN和MLP）之前平衡数据集，考虑了四种预处理策略（3种过采样和1种过采样），即SMOTE-TL，SPIDER，SMOTE-RSB *和安全级别的SMOTE。

[38]中考虑的乳腺癌检测案例与之前的案例相似，是良性分类的例子多于恶性分类的例子。同样，与遗漏癌症病例相关的成本比误贴癌症的成本要高得多。良性案例。在这项工作中，作者考虑了医学热成像技术。在热像图图像中，使用红外热像仪来研究感兴趣的区域。作者通过将成本敏感的决策树与同时执行FS和分类器融合的遗传算法相结合，解决了偏斜的类别分布问题。

[39]的作者没有专注于乳腺癌的检测，而是专注于对乳腺癌的恶性程度进行分级。这个问题是不平衡的，因为最高恶性等级是最重要的，即使它是病例数最少的恶性等级。为了处理不均匀的类分布，作者使用了EUSBoost，这是一种演化的欠采样增强方法，可生成大量的分类器。代替在增强过程中使用随机欠采样，可以考虑使用遗传算法来选择最合适的示例。

乳腺癌的检测也可以从磁共振图像（MRI）中进行[52]。这项工作提出了一种新的计算机辅助诊断系统，用于从MRI检测乳腺癌。在从MRI图像中提取特征阶段之后，使用一组随机欠采样的数据集解决了所得的不平衡数据集。每个新数据集都用于学习一个模型，其中FS和adaboost与C4.5决策树一起使用。

内窥镜或结肠镜检查图像中的息肉检测是ML可帮助临床医生的另一个问题。但是，由于息肉类型的多样性，检查成本高昂以及劳动密集的标记任务，息肉数据集往往是不平衡的，非息肉类的例子比息肉一类的例子多得多。在Bae和Yoon [6]的工作中，提出了一种基于数据采样的增强框架，以学习分类器来管理倾斜的类分布。所考虑的采样方案会生成综合的少数派类别示例（使用SMOTE），然后消除了难以对多数类别示例进行分类（使用Tomek Links）的情况。为了创建分类器的整体，将此重采样机制引入了AdaBoost算法。

胶囊内窥镜检查是传统有线内窥镜检查的患者友好替代品，在胃肠粘膜疾病的诊断和管理中起着重要作用。但是，在电池寿命期间，它们会生成数千张图像，需要临床医生花费大量精力进行分析。因此，在这项工作中，作者旨在开发一种能够检测这种内窥镜视频中出血的系统。由于图像很多，但很少出现出血，因此该问题会受到类别不平衡的困扰。作者研究了不同程度的失衡的影响。为了模拟这种情况，他们考虑了多数类的随机欠采样。测试了几种分类方法，例如SVM，ANN，决策树和KNN。他们还比较了这些方法与基于集成模型（RUSBoost）的鲁棒性。

糖尿病是所有人口和年龄组中最常见的内分泌疾病，是发达国家的主要死亡原因之一。

[15]的作者设计了一种系统来减轻对这种疾病的诊断，而阶级失衡起着重要的作用。与其他医疗问题一样，健康人比糖尿病人有更多例子。因此，开发了一种新的基于反向传播ANN的数据采样技术，然后将其结果数据集用于学习SVM分类器。

同样与糖尿病相关的是，微动脉瘤的检测对于降低患者由于进行性疾病糖尿病性视网膜病而致盲的可能性很重要。在[61]中，作者开发了一种用于微动脉瘤检测的计算机辅助检测系统，旨在减少先前工作中存在的假阳性数。为此，通过使用极端学习机作为基础分类器，结合不同的集成技术（Boosting，Bagging，RSM），通过对少数类进行过度采样（提出了自适应SMOTE）来解决类不平衡问题。

在其他方面，[16]的作者提出了一个问题，即预测在接下来的六个月中，第4阶段患有慢性肾脏疾病的患者是否会升至5级。这个问题是不平衡的，因为处于第4阶段的患者比进入第5阶段的患者更多。此类系统不仅使人们能够预测患者的未来状况，而且还有助于理解为什么患者可能会进展从一个阶段到下一个阶段。为了应对数据偏斜，作者考虑了随机欠采样的使用。之后，将使用平衡数据集来学习具有C4.5和CART决策树，SVM和Adaboost的模型。

#### 2.2.4.3 医疗预测

关于医学预测，我们已将两类作品归类。 [5]的作者进行了一项关于过采样和欠采样技术的研究，以解决他们的数据中存在的类别不平衡问题，从而改善对可能患有骨质疏松症的患者的筛查。为此，他们收集了将近1000名妇女的数据，目的是预测她们是否遭受过骨折。该系统使他们能够更好地选择可能患有骨质疏松症的患者。由于没有骨折的人比没有骨折的人少，因此作者对几种数据级技术进行了研究（随机欠采样，编辑最近邻和具有不同程度过采样程度的SMOTE）。预处理之后，学习了一些分类器，例如C4.5决策树，朴素贝叶斯，KNN和其他集成模型（Random Forest，Bagging和AdaBoost）。

在Prez-Ortiz等人的工作中。 [58]，提出了一种新的肝移植供体-受体分配系统。为了构建这样一个复杂的系统，作者开发了一个预测模型，其中预测了移植后的移植物存活率。以这种方式，不仅考虑了等待移植的患者的严重程度，而且考虑了这种移植是否成功。由于他们记录的成功移植案例比未成功结束的案例更多，因此类别分配存在偏差。为了解决这个问题，提出了一种基于生成数据的新方法（可以将其视为数据级方法），但是在这种情况下，生成的数据没有标签，因此，他们采用无监督分类解决了该问题。

### 2.2.5 业务管理

在这一类别中，我们收集了两种不同的用例，其中类的分布成为一个需要考虑的问题。一方面，桑兹等。 [64]处理了一些财务问题，其中该问题影响了模糊模型的准确性和可解释性。他们设计了一种新的基于区间值的模糊规则分类系统，该系统能够处理不平衡数据而无需更改提取的规则。新系统能够改善诸如股票市场预测，信用卡/贷款审批和欺诈检测等手头问题的性能。

另一方面，客户关系管理，更具体地说是客户流失预测，是另一个重要的用例，其中发现了偏斜的类分布。客户流失预测是不平衡的，因为大多数客户倾向于留在同一家公司。但是，检测客户流失对于改善与客户的关系并确定合适的客户来保留很重要。在[3]中，对流失预测的六种数据采样技术和四种不同的规则归纳方法进行了比较。同样，在[89]中，针对类不平衡问题进行了包括抽样，成本敏感和整体解决方案的综合研究。

### 2.2.6 安全性

类分布不平衡的大多数安全应用程序都可以在生物识别和视频监控中找到。 Radtke等人的工作解决了人脸识别问题。 [60]。在视频监控应用中，人脸识别用于检测复杂和不断变化的场景中感兴趣的目标个人的存在。问题在于，通常在学习时几乎没有可用的参考目标数据，这会产生不良的类别不平衡问题。作者提出了一种新方法，该方法基于修改偏斜敏感布尔组合方案如何将基本分类器组合在一起。还将该新方法与两种欠采样机制（随机欠采样和一侧选择）的性能进行了比较。他们的工作重点是面部重新识别，包括将实时或存档视频流中捕获的面部区域与已注册系统的个人的面部模型进行匹配。

相同的人脸识别问题在[68]中得到了解决。在这种情况下，作者提出了一种新的方法来计算Adaboost中的损失因子，并将其应用于解决类不平衡问题的基于集成的现有方法。同一作者开发了一种专门设计的Boosting方法，称为Progressive Boosting [69]，以提高人脸重新识别的分类性能。这种方法基于逐步将不相关的示例组插入到Boosting过程中，这可能是由于他们正在处理的应用程序类型而引起的。

### 2.2.7 教育

教育性DM是最近的研究领域，其中DM技术用于改善对社会的重要服务。在这一领域，类别分布不均时也存在问题，例如，检测早期辍学的问题[53]。在这项工作中，课程开始时从学生那里收集到的一些数据被用来学习一个模型，该模型能够检测学生在课程期间是否会离开学校。为了建立一个可解释的基于规则的模型，他们提出了使用基因编程的方法，其中对适应度函数进行了修改，以便解决问题固有的失衡（大多数学生不会辍学）。此外，将他们的算法与通过SMOTE进行的过采样进行了比较。

## 2.3 不平衡分类的案例研究

任何DM过程的最终目标都是应用于现实生活中的问题。由于在每个现有问题中测试一种技术都是不可行的，因此通用程序是在一组公开可用的DM问题（或数据集）中评估该技术。在这种情况下，我们专注于类不平衡问题，因此，这些数据集应具有此属性。在本节中，我们将介绍最常用的数据集，以比较为解决类不平衡问题而开发的方法。

用于不平衡数据集的最著名的存储库是KEEL数据集存储库[2]，它支持KEEL DM工具（将在本书中广泛使用）。在该存储库中可以找到的数据集是从不同的知名来源收集的，例如UCI存储库[46]。为了促进提案之间的比较，为数据集提供了交叉验证分区，这有助于减少可归因于不同分区方案的使用的算法之间的差异。此外，在此存储库中，有专门的部分专门讨论不平衡的问题。实际上，这是我们感兴趣的一种。

|||
|-|-|
|||
**表2.2 KEEL数据集存储库中的基准数据集**

为了获得两类不平衡问题，最初对多类数据集进行了修改，以使一个或多个类的联合成为正类，而其余一个或多个类的联合被标记为负类。这样，获得了具有不同IR的数据集：从低失衡数据集到高度失衡的数据集。表2.2总结了此存储库中数据集的属性，包括有关数据集的最相关信息：

- **数据集的名称:** 它编码如何获取数据集。例如，pima数据集本来是两类不平衡数据集，因此它没有经历任何变换。但是，名称glass1表示将类别1用作少数类别，而将其余类别的并集用作多数类别。类似地，在酵母2_vs_4的情况下，仅以一对一的方式考虑原始多类数据集中的两个类。其余数据集使用相同的命名法。
- **\#atts:** (R/I/N）是问题中的属性/特征及其类型的数量。 R代表实数值属性的数量，I代表整数属性的数量，N代表名义属性的数量。
- #**Ex:** 是数据集中的示例/实例数。
- **#Min:** 指示数据集中的少数类示例的数量。。
- **#Maj:** 是数据集中多数类示例的数量。
- IR是数据集中的不平衡率。


所有数据集均可在相应的网页上公开获得。

为了完整起见，列出了可用数据集的完整列表（表2.2）。请注意，该表分为四个组。第一个对应于不平衡比在1.5到9之间的数据集，它们被认为具有中等程度的不平衡度。其余组对应于不平衡比大于9的数据集，但由于它们在不同的时间可用，所以它们是分开的。实际上，最后一个对应于KEEL数据集存储库的最新添加，并且尚未在文献中广泛使用。

我们应该记得，本节中列出的所有数据集都是只有两个类的二进制数据集。 KEEL数据集存储库中还有一个专门介绍多类不平衡数据集的部分，该部分在相应的章节中介绍，即第8章。

除了KEEL数据集存储库外，还有另一个补充的基准集，其中包含较少的数据集。 HDDT collection2包含20个二进制不平衡数据集，这些数据集最初在[17]中用于验证所提出的模型。这些数据集中的大多数也是多类数据集的转换。表2.3总结了这些数据集的属性，遵循与表2.2中相同的方案。

比较KEEL数据集存储库和HDDT集合，前者是文献中用于算法比较的最广泛的方法，例如，参见[21-23、27、28、51]。无论如何，HDDT收集也已经在一些作品中使用[17，21，22]。
除了这些数据集外，还有另一个存储库，其中包含几个特定的数据集，用于软件开发中的缺陷预测，我们已经在Sect中审查了该应用程序。
2.2.2。这些数据集最初是由NASA度量数据程序提供的。但是，我们没有报告它们的完整细节，因为关于这些数据集的不同版本存在一些争议，可以在文献中找到[67]以及它们遭受的滥用[31]。也就是说，没有一个明确的基准测试，其中所有算法都使用完全相同的数据集进行测试。相同数据集存在不同版本的预处理，因此很难将它们用作通用基准。人们应该仔细使用这些数据集并注意其特定属性。无论如何，由于它们可能是该领域新方法的分析或开发所感兴趣的，因此，我们请读者参考当前的[55]和旧的[66] PROMISE数据集存储库，可以在其中找到这些数据集的不同版本。


1. Acharya, U.R., Chowriappa, P., Fujita, H., Bhat, S., Dua, S., Koh, J.E.W., Eugene, L.W.J.,Kongmebhol, P., Ng, K.: Thyroid lesion classification in 242 patient population using gabortransform features from high resolution ultrasound images. Knowl. Based Syst. 107, 235–245(2016)
2. Alcalá-Fdez, J., Fernández, A., Luengo, J., Derrac, J., García, S., Sánchez, L., Herrera,F.: KEEL data–mining software tool: data set repository, integration of algorithms andexperimental analysis framework. J. Multi–Valued Logic Soft Comput. 17(2–3), 255–287(2011)
3. Amin, A., Anwar, S., Adnan, A., Nawaz, M., Howard, N., Qadir, J., Hawalah, A., Hussain, A.:Comparing oversampling techniques to handle the class imbalance problem: a customer churnprediction case study. IEEE Access 4, 7940–7957 (2016)
4. Azari, A., Janeja, V.P., Levin, S.: Imbalanced learning to predict long stay emergencydepartment patients. In: IEEE International Conference on Bioinformatics and Biomedicine(BIBM), Washington, DC, pp. 807–814 (2015)
5. Bach, M., Werner, A., Z ́ ywiec, J., Pluskiewicz, W.: The study of under- and over-samplingmethods utility in analysis of highly imbalanced data on osteoporosis. Inf. Sci. 384, 174–190(2017)
6. Bae, S.H., Yoon, K.J.: Polyp detection via imbalanced learning and discriminative featurelearning. IEEE Trans. Med. Imaging 34(11), 2379–2393 (2015)
7. Barandela, R., Sánchez, J.S., García, V., Rangel, E.: Strategies for learning in class imbalanceproblems. Pattern Recogn. 36(3), 849–851 (2003)
8. Bashbaghi, S., Granger, E., Sabourin, R., Bilodeau, G.A.: Dynamic ensembles of exemplar-svms for still-to-video face recognition. Pattern Recogn. 69, 61–81 (2017)
9. Batista, G.E.A.P.A., Prati, R.C., Monard, M.C.: A study of the behavior of several methods forbalancing machine learning training data. SIGKDD Explor. Newslett. 6, 20–29 (2004)
10. Bermejo, P., Gámez, J.A., Puerta, J.M.: Improving the performance of naive bayes multinomialin e-mail foldering by introducing distribution-based balance of datasets. Expert Syst. Appl.38(3), 2072–2080 (2011)
11. Cao, P., Yang, J., Li, W., Zhao, D., Zaiane, O.: Ensemble-based hybrid probabilistic samplingfor imbalanced data learning in lung nodule CAD. Comput. Med. Imaging Graph. 38(3), 137–150 (2014)
12. Chawla, N.V., Japkowicz, N., Kolcz, A. (eds.): Special issue on learning from imbalanceddatasets. ACM SIGKDD Explor. Newslett. 6(1), 1–6 (2004)
13. Chawla, N., Cieslak, D., Hall, L., Joshi, A.: Automatically countering imbalance and itsempirical relationship to cost. Data Min. Knowl. Disc. 17, 225–252 (2008)
14. Chen, L.S., Cai, S.J.: Neural-network-based resampling method for detecting diabetes mellitus.J. Med. Biol. Eng. 35(6), 824–832 (2015)
15. Chen, Z., Yan, Q., Han, H., Wang, S., Peng, L., Wang, L., Yang, B.: Machine learning basedmobile malware detection using highly imbalanced network traffic. Inf. Sci. 433–434, 346–364(2018)
16. Cheng, L.C., Hu, Y.H., Chiou, S.H.: Applying the temporal abstraction technique to theprediction of chronic kidney disease progression. J. Med. Syst. 41(5), 85 (2017)
17. Cieslak, D.A., Hoens, T.R., Chawla, N.V., Kegelmeyer, W.P.: Hellinger distance decision treesare robust and skew-insensitive. Data Min. Knowl. Disc. 24(1), 136–158 (2012)
18. Dai, H.L.: Imbalanced protein data classification using ensemble FTM-SVM. IEEE Trans.NanoBiosci. 14(4), 350–359 (2015)
19. Dean, J., Ghemawat, S.: Mapreduce: simplified data processing on large clusters. Commun.ACM 51(1), 107–113 (2008)
20. Deeba, F., Mohammed, S.K., Bui, F.M., Wahid, K.A.: An empirical study on the effect ofimbalanced data on bleeding detection in endoscopic video. In: 38th Annual InternationalConference of the IEEE Engineering in Medicine and Biology Society (EMBC), Orlando,pp. 2598–2601 (2016)
21. Díez-Pastor, J.F., Rodríguez, J.J., García-Osorio, C., Kuncheva, L.I.: Random balance: ensem-bles of variable priors classifiers for imbalanced data. Knowl. Based Syst. 85, 96–111 (2015)
22. Díez-Pastor, J.F., Rodríguez, J.J., García-Osorio, C.I., Kuncheva, L.I.: Diversity techniquesimprove the performance of the best imbalance learning ensembles. Inf. Sci. 325, 98–117(2015)
23. Fernández, A., García, S., del Jesus, M.J., Herrera, F.: A study of the behaviour of linguisticfuzzy rule based classification systems in the framework of imbalanced data–sets. Fuzzy SetsSyst. 159(18), 2378–2398 (2008)
24. Fernández, A., García, S., del Jesus, M.J., Herrera, F.: A study of the behaviour of linguisticfuzzy rule based classification systems in the framework of imbalanced data-sets. Fuzzy SetsSyst. 159(18), 2378–2398 (2008)
25. Fernández-Navarro, F., Hervás-Martínez, C., Gutiérrez, P.A.: A dynamic over-sampling proce-dure based on sensitivity for multi-class problems. Pattern Recogn. 44(8), 1821–1833 (2011)
26. Galar, M., Fernández, A., Barrenechea, E., Bustince, H., Herrera, F.: An overview of ensemblemethods for binary classifiers in multi-class problems: experimental study on one-vs-one andone-vs-all schemes. Pattern Recogn. 44(8), 1761–1776 (2011)
27. Galar, M., Fernández, A., Barrenechea, E., Bustince, H., Herrera, F.: A review on ensemblesfor class imbalance problem: bagging, boosting and hybrid based approaches. IEEE Trans.Syst. Man Cybern. Part C Appl. Rev. 42(4), 463–484 (2012)
28. Galar, M., Fernández, A., Barrenechea, E., Herrera, F.: Eusboost: enhancing ensembles forhighly imbalanced data-sets by evolutionary undersampling. Pattern Recogn. 46(12), 3460–3471 (2013)
29. García, V., Mollineda, R., Sánchez, J.: On the k-nn performance in a challenging scenario ofimbalance and overlapping. Pattern. Anal. Appl. 11, 269–280 (2008)
30. Garg, S., Sarje, A.K., Peddoju, S.K.: Improved detection of p2p botnets through networkbehavior analysis. In: Martínez Pérez, G., Thampi, S.M., Ko, R., Shu, L. (eds.) Recent Trendsin Computer Networks and Distributed Systems Security: Second International Conference,SNDS 2014, Trivandrum, 13–14 Mar 2014, Proceedings, pp. 334–345. Springer, Berlin/Hei-delberg (2014)
31. Gray, D., Bowes, D., Davey, N., Sun, Y., Christianson, B.: The misuse of the nasa metrics dataprogram data sets for automated software defect prediction. In: 15th Annual Conference onEvaluation Assessment in Software Engineering (EASE 2011), Durham, pp. 96–103 (2011)
32. Haddad, B.M., Yang, S., Karam, L.J., Ye, J., Patel, N.S., Braun, M.W.: Multifeature, sparse-based approach for defects detection and classification in semiconductor units. IEEE Trans.Autom. Sci. Eng. 15(1), 144–159 (2017)
33. Haixiang, G., Yijing, L., Shang, J., Mingyun, G., Yuanyue, H., Bing, G.: Learning from class-imbalanced data: review of methods and applications. Expert Syst. Appl. 73, 220–239 (2017)
34. He, H., Garcia, E.A.: Learning from imbalanced data. IEEE Trans. Knowl. Data Eng. 21(9),1263–1284 (2009)
35. Iannello, G., Percannella, G., Soda, P., Vento, M.: Mitotic cells recognition in HEp-2 images.Pattern Recogn. Lett. 45, 136–144 (2014)
36. Japkowicz, N., Stephen, S.: The class imbalance problem: a systematic study. Intell. Data Anal.6, 429–449 (2002)
37. Khreich, W., Granger, E., Miri, A., Sabourin, R.: Iterative boolean combination of classifiersin the ROC space: an application to anomaly detection with hmms. Pattern Recogn. 43(8),2732–2752 (2010)
38. Krawczyk, B., Schaefer, G., Woz ́niak, M.: A hybrid cost-sensitive ensemble for imbalanced breast thermogram classification. Artif. Intell. Med. 65(3), 219–227 (2015)
39. Krawczyk, B., Galar, M., Jelen ́, L., Herrera, F.: Evolutionary undersampling boosting forimbalanced classification of breast cancer malignancy. Appl. Soft Comput. 38, 714–726 (2016)
40. Kubat, M., Matwin, S.: Addressing the curse of imbalanced training sets: one-sided selection.In: Proceedings of the 14th International Conference on Machine Learning, pp. 179–186. Morgan Kaufmann, San Francisco (1997)
41. Kubat, M., Holte, R., Matwin, S.: Learning when negative examples abound. In: van Someren, M., Widmer, G. (eds.) Proceedings of the 9th European Conference on Machine Learning,pp. 146–153. Springer, Berlin/Heidelberg (1997)
42. Kubat, M., Holte, R.C., Matwin, S.: Machine learning for the detection of oil spills in satellite radar images. Mach. Learn. 30(2), 195–215 (1998)
43. Laradji, I.H., Alshayeb, M., Ghouti, L.: Software defect prediction using ensemble learning on selected features. Inf. Softw. Technol. 58, 388–402 (2015)
44. Lee, T., Lee, K.B., Kim, C.O.: Performance of machine learning algorithms for class-imbalanced process fault detection problems. IEEE Trans. Semicond. Manuf. 29(4), 436–445(2016)
45. Lertampaiporn, S., Thammarongtham, C., Nukoolkit, C., Kaewkamnerdpong, B., Ruengjitchatchawalya, M.: Heterogeneous ensemble approach with discriminative featuresand modified-smotebagging for pre-mirna classification. Nucleic Acids Res. 41(1), e21 (2013)
46. Lichman, M.: UCI machine learning repository. School of Information and Computer Sciences,University of California, Irvine (2013). http://archive.ics.uci.edu/ml
47. Lin, Y., Lee, Y., Wahba, G.: Support vector machines for classification in nonstandardsituations. Mach. Learn. 46, 191–202 (2002)
48. Ling, C., Sheng, V., Yang, Q.: Test strategies for cost-sensitive decision trees. IEEE Trans.Knowl. Data Eng. 18(8), 1055–1067 (2006)
49. Liu, Y.H., Chen, Y.T.: Total margin based adaptive fuzzy support vector machines for multiviewface recognition. In: IEEE International Conference on Systems, Man and Cybernetics,Waikoloa, vol. 2, pp. 1704–1711 (2005)
50. Liu, B., Ma, Y., Wong, C.: Improving an association rule based classifier. In: Zighed, D.,Komorowski, J., Zytkow, J. (eds.) Principles of Data Mining and Knowledge Discovery. LNCS,vol. 1910, pp. 293–317. Springer, Berlin/Heidelberg (2000)
51. López, V., Fernández, A., García, S., Palade, V., Herrera, F.: An insight into classification withimbalanced data: empirical results and current trends on using data intrinsic characteristics.Inf. Sci. 250(20), 113–141 (2013)
52. Lu, W., Li, Z., Chu, J.: A novel computer-aided diagnosis system for breast {MRI} based onfeature selection and ensemble learning. Comput. Biol. Med. 83, 157–165 (2017)
53. Márquez-Vera, C., Cano, A., Romero, C., Noaman, A.Y.M., Mousa Fardoun, H., Ventura, S.:Early dropout prediction using data mining: a case study with high school students. ExpertSyst. 33(1), 107–124 (2016)
54. Mazurowski, M.A., Habas, P.A., Zurada, J.M., Lo, J.Y., Baker, J.A., Tourassi, G.D.: Trainingneural network classifiers for medical decision making: the effects of imbalanced datasets onclassification performance. Neural Netw. 21(2–3), 427–436 (2008)
55. Menzies, T., Krishna, R., Pryor, D.: The promise repository of empirical software engineeringdata. Department of Computer Science, North Carolina State University (2015). http://www.openscience.us/repo
56. Napierała, K., Stefanowski, J., Wilk, S.: Learning from imbalanced data in presence of noisyand borderline examples. In: Kryszkiewicz, M., et al. (eds.) Rough Sets and Current Trends inComputing, pp. 158–167. Springer, Berlin/Heidelberg (2010)
57. Orriols-Puig, A., Bernadó-Mansilla, E.: Evolutionary rule-based systems for imbalanced datasets. Soft Comput. 13, 213–225 (2009)
58. Pérez-Ortiz, M., Gutiérrez, P., Ayllón-Terán, M., Heaton, N., Ciria, R., Briceño, J., Hervás-Martínez, C.: Synthetic semi-supervised learning in imbalanced domains: constructing a modelfor donor-recipient matching in liver transplantation. Knowl. Based Syst. 123, 75–87 (2017)
59. Polikar, R.: Ensemble based systems in decision making. IEEE Circuits Syst. Mag. 6(3), 21–45(2006)
60. Radtke, P.V., Granger, E., Sabourin, R., Gorodnichy, D.O.: Skew-sensitive boolean combina-tion for adaptive ensembles – an application to face recognition in video surveillance. Inf.Fusion 20, 31–48 (2014)
61. Ren, F., Cao, P., Li, W., Zhao, D., Zaiane, O.: Ensemble based adaptive over-sampling methodfor imbalanced data learning in computer aided detection of microaneurysm. Comput. Med.Imaging Graph. 55, 54–67 (2017). Special Issue on Ophthalmic Medical Image Analysis
62. Rodriguez, D., Herraiz, I., Harrison, R., Dolado, J., Riquelme, J.C.: Preliminary comparisonof techniques for dealing with imbalance in software defect prediction. In: Proceedings of the18th International Conference on Evaluation and Assessment in Software Engineering, EASE’14, pp. 43:1–43:10. ACM, New York (2014)
63. Santos, P., Maudes, J., Bustillo, A.: Identifying maximum imbalance in datasets for faultdiagnosis of gearboxes. J. Intell. Manuf. 29(2), 333–351 (2018)
64. Sanz, J.A., Bernardo, D., Herrera, F., Bustince, H., Hagras, H.: A compact evolutionaryinterval-valued fuzzy rule-based classification system for the modeling and prediction of real-world financial applications with imbalanced data. IEEE Trans. Fuzzy Syst. 23(4), 973–990(2015)
65. Sanz, J., Fernandez, J., Bustince, H., Gradin, C., Fortun, M., Belzunegui, T.: A decision treebased approach with sampling techniques to predict the survival status of poly-trauma patients.Int. J. Comput. Intell. Syst. 10(1), 440–455 (2017)
66. Sayyad Shirabad, J., Menzies, T.: The PROMISE repository of software engineering databases.School of Information Technology and Engineering, University of Ottawa, Canada (2005).http://promise.site.uottawa.ca/SERepository
67. Shepperd, M., Song, Q., Sun, Z., Mair, C.: Data quality: some comments on the nasa softwaredefect datasets. IEEE Trans. Softw. Eng. 39(9), 1208–1215 (2013)
68. Soleymani, R., Granger, E., Fumera, G.: Loss factors for learning boosting ensembles fromimbalanced data. In: 23rd International Conference on Pattern Recognition (ICPR), Cancún,pp. 204–209 (2016)
69. Soleymani, R., Granger, E., Fumera, G.: Progressive boosting for class imbalance and itsapplication to face re-identification. Expert Syst. Appl. 101, 271–291 (2018)
70. Stefanowski, J., Wilk, S.: Selective pre-processing of imbalanced data for improving classifica-tion performance. In: Song, I.Y., Eder, J., Nguyen, T. (eds.) Data Warehousing and KnowledgeDiscovery. LNCS, vol. 5182, pp. 283–292. Springer, Berlin/Heidelberg (2008)
71. Sun, Y., Wong, A.C., Kamel, M.S.: Classification of imbalanced data: a review. Int. J. PatternRecognit. Artif. Intell. 23(4), 687–719 (2009)
72. Sun, Z., Song, Q., Zhu, X.: Using coding-based ensemble learning to improve software defectprediction. IEEE Trans. Syst. Man Cybern. Part C Appl. Rev. 42(6), 1806–1817 (2012)
73. Tang, M., Ding, S.X., Yang, C., Cheng, F., Shardt, Y.A.W., Long, W., Liu, D.: Cost-sensitivelarge margin distribution machine for fault detection of wind turbines. Clust. Comput. 1–13(2018). https://doi.org/10.1007/s10586-018-1854-3
74. Tavallaee, M., Stakhanova, N., Ghorbani, A.: Toward credible evaluation of anomaly-basedintrusion-detection methods. IEEE Trans. Syst. Man Cybern. Part C Appl. Rev. 40(5), 516–524 (2010)
75. Tax, D.M.J.: One-class classification: concept learning in the absence of counter-examples.Ph.D. thesis, Technische Universiteit Delft (2001)
76. Tomek, I.: Two modifications of CNN. IEEE Trans. Syst. Man Cybern. SMC-6(11), 769–772(1976)
77. Tran, Q.D., Liatsis, P.: Raboc: an approach to handle class imbalance in multimodal biometricauthentication. Neurocomputing 188, 167–177 (2016). Advanced Intelligent ComputingMethodologies and Applications
78. Triguero, I., del Río, S., López, V., Bacardit, J., Benítez, J.M., Herrera, F.: ROSEFW-RF: thewinner algorithm for the ECBDL–14 big data competition: An extremely imbalanced big databioinformatics problem. Knowl. Based Syst. 87, 69–79 (2015). Computational IntelligenceApplications for Data Science
79. Wang, S., Yao, X.: Multiclass imbalance problems: analysis and potential solutions. IEEETrans. Syst. Man Cybern. B Cybern. 42(4), 1119–1130 (2012)
80. Wei, H., Sun, B., Jing, M.: Balancedboost: a hybrid approach for real-time network trafficclassification. In: 23rd International Conference on Computer Communication and Networks(ICCCN), Shanghai, pp. 1–6 (2014)
81. Weiss, G.M., Provost, F.: Learning when training data are costly: the effect of class distributionon tree induction. J. Artif. Intell. Res. 19, 315–354 (2003)
82. Wu, Q., Ye, Y., Zhang, H., Ng, M.K., Ho, S.S.: Forestexter: an efficient random forest algorithmfor imbalanced text categorization. Knowl. Based Syst. 67, 105–116 (2014)
83. Wu, Z., Lin, W., Ji, Y.: An integrated ensemble learning model for imbalanced fault diagnosticsand prognostics. IEEE Access 6, 8394–8402 (2018)
84. Yang, Z., Tang, W., Shintemirov, A., Wu, Q.: Association rule mining-based dissolved gasanalysis for fault diagnosis of power transformers. IEEE Trans. Syst. Man Cybern. Part CAppl. Rev. 39(6), 597–610 (2009)
85. Yang, P., Yoo, P.D., Fernando, J., Zhou, B.B., Zhang, Z., Zomaya, A.Y.: Sample subsetoptimization techniques for imbalanced and ensemble learning problems in bioinformaticsapplications. IEEE Trans. Cybern. 44(3), 445–455 (2014)
86. Zhang, S., Liu, L., Zhu, X., Zhang, C.: A strategy for attributes selection in cost-sensitivedecision trees induction. In: IEEE 8th International Conference on Computer and InformationTechnology Workshops, Sydney, pp. 8–13 (2008)
87. Zhang, X., Zhuang, Y., Wang, W., Pedrycz, W.: Transfer boosting with synthetic instances forclass imbalanced object recognition. IEEE Trans. Cybern. 48(1), 357–370 (2018)
88. Zhu, Z.B., Song, Z.H.: Fault diagnosis based on imbalance modified kernel fisher discriminantanalysis. Chem. Eng. Res. Des. 88(8), 936–951 (2010)
89. Zhu, B., Baesens, B., vanden Broucke, S.K.L.M.: An empirical comparison of techniques forthe class imbalance problem in churn prediction. Inf. Sci. 408, 84–99 (2017)
90. Zhu, L., Lu, C., Dong, Z.Y., Hong, C.: Imbalance learning machine based power system short-term voltage stability assessment. IEEE Trans. Ind. Inf. 13(5), 2533–2543 (2017)
91. Zieba, M., Tomczak, J.M., Lubicz, M., S ́wiatek, J.: Boosted SVM for extracting rules fromimbalanced data in application to prediction of the post-operative life expectancy in the lungcancer patients. Appl. Soft Comput. 14, Part A, 99–108 (2014)


  [1]: https://raw.githubusercontent.com/youfeng8/aidata_wiki/master/image/【learningfromimbalanceddatasets】第二章、不平衡分类的基础/image_1etc2u4841rrpksd1djnjkceir9.png
  [2]: https://raw.githubusercontent.com/youfeng8/aidata_wiki/master/image/【learningfromimbalanceddatasets】第二章、不平衡分类的基础/image_1etc2un5hkr61osp1vos1ioj1eipm.png
  [3]: https://raw.githubusercontent.com/youfeng8/aidata_wiki/master/image/【learningfromimbalanceddatasets】第二章、不平衡分类的基础/image_1etc30o8r16co1gvkms7svaah813.png