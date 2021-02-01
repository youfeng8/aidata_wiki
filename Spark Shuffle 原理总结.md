# Spark Shuffle 原理总结

标签（空格分隔）： 大数据挖掘与分析

---

Shuffle是连接Map和Reduce之间的桥梁。在Shuffle操作中，负责分发数据的Executor叫做Mapper，而接收数据的Executor叫做Reducer。

因为在分布式情况下，Reduce task需要跨节点去拉取其它节点上的Map task结果。过程将会消耗网络，内存，磁盘IO，CPU等。

通常shuffle分为两部分：Map阶段的数据准备和Reduce阶段的数据拷贝处理。



## 一、回顾：

**hadoop Shuffle：**

- **Map端:**
1. input, 根据split输入数据，运行map任务;
2. patition, 每个map task都有一个内存缓冲区，存储着map的输出结果;
3. spill, 当缓冲区快满的时候需要将缓冲区的数据以临时文件的方式存放到磁盘;
4. merge, 当整个map task结束后再对磁盘中这个map task产生的所有临时文件做合并，生成最终的正式输出文件，然后等待reduce task来拉数据。
    ["北京", ["朝阳","密云","顺义","昌平"]]

- **Reduce端:**
reduce task在执行之前的工作就是不断地拉取当前job里每个map task的最终结果，然后对从不同地方拉取过来的数据不断地做merge，也最终形成一个文件作为reduce task的输入文件。
1. Copy过程，拉取数据; (网络瓶颈)
2. Merge阶段，合并拉取来的小文件;
3. Reducer计算;
4. Output输出计算结果

## 二、Spark Shuffle

以上是hadoop Shuffle， 接下来介绍Spark Shuffle

Shuffle时，有两个很重要的用于压缩的参数:

- spark.shuffle.compress – 是否要Spark对Shuffle的输出进行压缩
- spark.shuffle.spill.compress – 是否压缩Shuffle中间的刷写到磁盘的文件

这两个参数默认都是true。并且，默认都会使用spark.io.compression.codec来压缩数据，默认情况下，是snappy

shuffle类型取决于spark.shuffle.manager这个参数，主要分为Hash, Sort和Tungsten-Sort从Spark 1.2.0开始，sort是默认选项。

###2.1、Hash

设置参数：**spark.shuffle.manager = hash**

#### 2.1.1 v1 版本

在 Map Task过程按照Hash的方式重组Partition的数据，不进行排序。每个MapTask为每个ReduceTask生成一个文件，通常会产生大量的文件（即对应为M*R个中间文件，其中M表示MapTask个数，R表示ReduceTask个数），伴随大量的随机磁盘 I/O 操作与大量的内存开销。

![image_1eter7ns01gep1qve1s051rsoah9.png-167.8kB][1]

**缺点:**

- 生成大量文件，占用文件描述符，同时引入 DiskObjectWriter 带来的 Writer Handler 的缓存也非常消耗内存；
- 如果在 Reduce Task 时需要合并操作的话，会把数据放在一个 HashMap 中进行合并，如果数据量较大，很容易引发 OOM。

#### 2.1.2 v2 版本

针对上面的第一个问题，Spark做了改进，引入了**File Consolidation**机制。

一个Executor上所有的MapTask生成的分区文件只有一份，即将所有的MapTask相同的分区文件合并，这样每个Executor上最多只生成N个分区文件。

![image_1eter9ol86hn10dj1d0vibd1kmh13.png-225.7kB][2]

**缺点:** Executor上有K个Core，还是会开K*N个WriterHandler，所以这里仍然容易导致OOM。

### 2.2、Sort

设置参数：**spark.shuffle.manager = sort**

Spark参考了MapReduce中Shuffle的处理方式，引入基于排序的Shuffle写操作机制。每个Task不会为后续的每个Task创建单独的文件，而是将所有对结果写入同一个文件。

![image_1eterchvc1c7tnc1190u68b1sf01g.png-569.4kB][3]
**Map:**

- 该文件中的记录首先是按照PartitionId排序，每个Partition内部再按照Key进行排序，
- 再分批写入磁盘文件，分批的batch数量大小为1w条，最后将产生的多个磁盘文件merge成一个磁盘文件，并产生一个索引文件，用以标识下游stage中的各个task的数据在文件中的start offset 和 end offset，直观来看，**一个task仅产生一个磁盘文件和一个索引文件**

**Reduce:**
在Reduce阶段，ReduceTask拉取数据做Combine时不再是采用HashMap，而是采用ExternalAppendOnlyMap，该数据结构在做Combine时，如果内存不足，会刷写磁盘，很大程度的保证了鲁棒性，避免大数据情况下的OOM。

SortShuffle也分为**普通机制和bypass机制**，普通机制在内存数据结构(默认为5M)完成排序，会产生2M个磁盘小文件。而当shuffle map task数量小于spark.shuffle.sort.bypassMergeThreshold参数的值。或者算子不是聚合类的shuffle算子(比如reduceByKey)的时候会触发SortShuffle的bypass机制，SortShuffle的bypass机制不会进行排序，极大的提高了其性能

在ShuffleManager一路优化的过程中，**一个重要优化思想其实就是在减少shuffle过程中产生磁盘文件数量**，一个直观的逻辑：磁盘文件少，上下游stage需要进行的磁盘IO操作就相对少了。而磁盘文件过多会带来以下问题：

- 如果磁盘文件多，进行shuffle操作时需要同时打开的文件数多，大量的文件句柄和写操作分配的临时内存将对内存和GC带来压力，特别是在YARN的模式下，往往Executor分配的内存不足以支持这么大的内存压力；
- 如果磁盘文件多，那么其所带来的随机读写需要多次磁盘寻道和旋转延迟，比顺序读写的时间多许多倍。

**优点:**

- Mapper创建的文件少
- Random IO更少，大多数都是sequential writes和reads

**缺点:**

- 排序比Hash慢。所以需要细心找出bypassMergeThreshold。因为默认值可能有点大
- 如果你使用SSD，那么Hash shuffle可能更好(译者注：为啥?因为Random IO代价更小么？但是创建了很多文件的事情没办法解决啊)

### 2.3、tungsten-sort

设置参数：**spark.shuffle.manager = tungsten-sort**

**优化点：**

- 可以不经过反序列化而直接操纵数据。因为它内部使用的是unsafe (sun.misc.Unsafe) memory copy functions。
- 内部对CPU cache做了优化。它在排序时，每个record只使用8 bytes，因为把record pointer以及partition id做了压缩。对应的排序算法是 ShuffleExternalSorter
- 将数据刷到磁盘上时，不需要经过反序列化(no deserialize-compare-serialize-spill logic)。
- 如果压缩算法支持直接对序列化的stream进行拼接，那么，就可以使用spill-merge优化。现在只有Spark的LZF serializer支持直接对序列化之后的Stream进行拼接。并且只有在开起了shuffle.unsafe.fastMergeEnabled才起作用。

Shuffle条件：只有满足如下条件才会Shuffle

- 不是为了做聚集操作才做Shuffle。因为如果做聚集操作的话，很明显需要反序列化数据，并进行聚集。这样子的话，就失去了Unsafe Shuffle最重要的优势-直接对序列化后的数据进行操作。
- Shuffle serializer支持serialized values的重定位(译者注：没看懂)。当前只有KrySerializer以及Sparkr SQL的自定义serailizer支持。
- Shuffle产生的Partition数量，小于16777216
- 序列化之后，每条Record的大小，不能大于128MB

另外，有一点需要注意的是，在sort时，只会根据partition id 进行排序。也就是说，在Reducer端执行的merge pre-sorted data，以及依赖的TimSort算法，现在都将不起作用。

sort时，依靠8-byte values，每个value都会包含一个指向序列化以后数据的指针，以及partition id，这也是我们为什么说partition的数量不能超过16777216.(译者注：16777216是2**24，也就是说，这8-bytes里面，指针占5-byte，另外3-byte 才是partition id)

**优点：**

- 做了很多性能优化

**缺点:**

- 不会在Mapper端处理数据的顺序问题
- 没有提供堆外排序内存


## 参考：

- [Spark中的Spark Shuffle详解](https://www.cnblogs.com/itboys/p/9226479.html)
- [Spark Shuffle 详解](https://zhuanlan.zhihu.com/p/67061627)
- [Spark架构-Shuffle](https://www.jianshu.com/p/a3bb3001abae) 
- [Spark之shuffle原理及性能优化](https://www.jianshu.com/p/98a1d67bc226)

源码：
    https://github.com/apache/spark/tree/branch-2.2/core/src/main/scala/org/apache/spark/shuffle


  [1]: http://static.zybuluo.com/tc1052400205/nlrp81v5yc0hc5n28q8jw6dg/image_1eter7ns01gep1qve1s051rsoah9.png
  [2]: http://static.zybuluo.com/tc1052400205/4whwrnbzlkx395qq84jhy039/image_1eter9ol86hn10dj1d0vibd1kmh13.png
  [3]: http://static.zybuluo.com/tc1052400205/0m45rjnplmkb3pm2yl8ncskx/image_1eterchvc1c7tnc1190u68b1sf01g.png