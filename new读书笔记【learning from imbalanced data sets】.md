# 读书笔记【learning from imbalanced data sets】

标签（空格分隔）： AI_不平衡学习

---
![image_1etc0dgh810vvvp417e817n9rvj9.png-615.5kB][1]

## 绪论
使用不平衡数据进行学习是指这样一种场景，其中代表给定问题中的概念的实例数量遵循不同的分布。解决此类学习问题的主要问题是，每个班级获得的准确性也不同。发生这种情况是因为大多数分类算法的学习过程通常偏向多数类实例，因此少数类不能很好地建模到最终系统中。作为现实应用中非常常见的场景，这些年来，研究人员和从业人员对该主题的兴趣已显着增长。

基于作者在专注于不平衡分类的几年之后的经验，本书旨在为对这一研究领域感兴趣的任何人提供一个一般而可理解的概述。它包含对该问题的正式描述，并着重于其主要特征和最相关的建议解决方案。此外，它考虑了数据科学中不平衡分类可能会带来真正挑战的不同情况。

在第一章中对KDD流程和数据科学的现状进行了温和的介绍之后，本书通过建立基础并回顾了在第2章中直接应用于该领域的案例研究，强调了与标准分类任务之间的差距。然后，第3章介绍了该研究领域要考虑的主要临时评估指标。该书还介绍了传统上用于解决二进制偏斜类分布的各种方法。具体来说，它回顾了成本敏感型学习（第4章），数据级预处理方法（第5章）和算法级解决方案（第6章），同时还考虑了那些嵌入了任何内容的集成学习解决方案。以前的选择（第7章）。此外，它集中在第一章。关于多类问题的扩展问题的第8章，不再以简单的方式应用以前的经典方法。

该书在第9章中提供了一些有关数据缩减的说明，以了解与使用这种方法有关的优点。然后，第10章将重点放在数据固有特性上，这些特性是导致这种情况的主要原因，而这些固有特性会加重到不均匀的类分布中，从而真正阻碍了分类算法的性能。最后，这本书介绍了一些新颖的研究领域，这些领域正在引起对不平衡数据问题的更深入的关注。具体而言，第11章考虑了数据流的分类，第12章考虑了非经典的分类问题，最后第13章讨论了与大数据有关的可伸缩性。总而言之，第14章中给出了一些用于解决不平衡分类的软件库和模块的示例。

对不平衡分类的当前和将来状态进行的全面审查旨在使该主题具有应有的意义。尤其是，出版物和引文的逐年增加表明了研究和学术界的兴趣。在可预见的将来，由于必须从不平衡分类的角度来解决许多现代实际应用，因此可以预见的是它将以新颖的重大发展继续扩展。

本书的目标读者是致力于应用不平衡学习技术来解决各种现实问题的开发人员和工程师，以及需要对从不平衡数据中学习的技术，方法和工具进行全面审查的研究人员和学生。

**目录：**
1. Introduction to KDD and Data Science
2. Foundations on Imbalanced Classification
3. Performance Measures
4. Cost-Sensitive Learning
5. Data Level Preprocessing Methods
6. Algorithm-Level Approaches
7. Ensemble Learning
8. Imbalanced Classification with Multiple Classes
9. Dimensionality Reduction for lmbalanced Learning
10. Data Intrinsic Characteristics
11. Learning from lImbalanced Data Streams
12. Non-classical lmbalanced Classification Problems
13. Imbalanced Classification for Big Data
14. Software and Libraries for Imbalanced Classification

  [1]: https://raw.githubusercontent.com/youfeng8/aidata_wiki/master/image/读书笔记【learning from imbalanced data sets】/image_1etc0dgh810vvvp417e817n9rvj9.png