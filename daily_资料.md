

重磅！阿里开源自研语音识别模型DFSMN，准确率高达96.04%
阿里技术  作者： 张仕良  发布于 2018-06-08 10:59:59 举报 阅读数：2392
​​​阿里妹导读：近日，阿里巴巴达摩院机器智能实验室开源了新一代语音识别模型DFSMN，将全球语音识别准确率纪录提高至96.04%（这一数据测试基于世界最大的免费语音识别数据库LibriSpeech）。
对比目前业界使用最为广泛的LSTM模型，DFSMN模型训练速度更快、识别准确率更高。采用全新DFSMN模型的智能音响或智能家居设备，相比前代技术深度学习训练速度提到了3倍，语音识别速度提高了2倍。
开源地址：
https://github.com/tramphero/kaldi


另外Newton法真的比Gradient Descend快很多，但求解Hessian矩阵逆比较麻烦。不过对于最小二乘的线性拟合，Hessian是常矩阵所以才能快速求解

终于知道前几天弄的多元函数的牛顿迭代求最值的问题错在哪里了，原来是特么的整型转成浮点被截取了,MDZZ[doge] ​

微软宣布收购伯克利加州大学Dan Klein教授和斯坦福Percy Liang教授的对话系统创业公司Semantic Machines: O网页链接

『深度网络架构笔记』《Define some common neural network architectures and ideas》  GitHub：O网页链接 ​​​​

『机器学习研究人员需要了解的8个神经网络架构』《The 8 Neural Network Architectures Machine Learning Researchers Need to Learn》O网页链接

#TheDownload#【微软收购Semantic Machines，以强化Cortana自然对话能力】
今天微软宣布收购总部位于加州伯克利的Semantic Machines公司，该公司创建了革命性的新方法来建立会话式AI。
Semantic Machines的创始人都是会话式AI领域的先锋人物，包括技术企业家Dan Roth、两名全球最著名的自然语言AI研究专家——来自加州大学伯克利分校教授Dan Klein和斯坦福大学教授Percy Liang、以及前苹果首席语音专家Larry Gillick。收起全文d

对话系统的一个问题是，现在依然没有一个最好的对话管理方法论出现，方法论应用到实际案例也必须根据情况不同有不同程度的设计和改变。这些人为参与，大大的增加了整个系统的设计难度，也增加了人的学习成本。 ​​​​

【机器学习热门开源项目(2018.5)】《Machine Learning Open Source of the Month (v.May 2018)》by Mybridge O网页链接 pdf:O网页链接

至此，完成了博士期间的所有论文。过去5年，在人工智能顶级会议和期刊共发表25篇论文：5篇ICML, 1篇JMLR，3篇UAI，3篇ACL，2篇KDD，1篇ICCV，1篇NAACL，2篇IJCAI, 2篇AAAI，以及ATC, 中国工程院院刊，IEEE Big Data, ECML, Heredity各1篇。感谢导师Eric Xing教授的悉心指导，为我指明正确的方向，传授正确的方法论，帮助我培养好的研究品味，并给予很大的研究自由。感谢合作的各位老师和同学的大力帮助，很幸运能跟你们一起工作。

深度学习，反击虚假评论的一记重拳！在 Jupyter Notebook 中使用 Keras 和 Tensorflow 来训练一个深度学习语言模型。O网页链接 ​​​​


卷积神经网络中十大拍案叫绝的操作
一、卷积只能在同一组进行吗？– Group convolution 
二、卷积核一定越大越好？– 3×3卷积核 
三、每层卷积只能用一种尺寸的卷积核？– Inception结构 
四、怎样才能减少卷积层参数量？– Bottleneck 
五、越深的网络就越难训练吗？– Resnet残差网络 
六、卷积操作时必须同时考虑通道和区域吗？– DepthWise操作
七、分组卷积能否对通道进行随机分组？– ShuffleNet 
八、通道间的特征都是平等的吗？ – SEnet 
九、能否让固定大小的卷积核看到更大范围的区域？– Dilated convolution 
十、卷积核形状一定是矩形吗？– Deformable convolution 可变形卷积核 

【Building a Content-Based Search Engine IV: Earth Mover’s Distance】O网页链接 建立一个基于内容的搜索引擎IV: Earth Mover的距离。 

『深度学习最新方法：随机加权平均，击败了当前最先进的Snapshot Ensembling』《Stochastic Weight Averaging — a New Way to Get State of the Art Results in Deep Learning》O网页链接 ​​​​

“据我所知，如今的各种先进软件技术，从人工智能到虚拟现实，从计算机视觉到3D音频，其实都是矩阵乘法。要是孩子问你为什么非要在高中学线性代数，这就是原因。对长除法我可没有这么巧妙的回答。” via:Chris Anderson ​​​​

pullword全面京东云化了，谢谢京东云支持和赞助了在线分词业务，谢谢啊。也不知道靠化缘，这个东西能走多远。总之这一年是不愁吃穿了，真好啊。正好我也测试下京东云的稳定性，给大家当小白鼠了。 ​​​​

2.23-2.28学习TensorFlow官方文档中文版
    178页 共6天 每天30页 每个早晨 中午 晚上 10页

腾讯犀牛鸟学问
    清华-腾讯联合实验室的技术交流
    
    常铭珊@将门创投-at_id:6 发言 ：
    明晚8点@将门斗鱼直播间，三角兽科技首席科学家王宝勋博士，会为我们分享自动聊天系统的技术路线，解决方案与挑战
    欢迎大家搬好小板凳听nlp届相声艺术家妙语连珠


“Deep Learning Indaba 2017 Tutorials, Video And Slides” O网页链接 ​​​​

Phrase-Based & Neural Unsupervised Machine Translation   O网页链接    无监督能干的事情那就更多了 ​

(Doctor Thesis)《Approaches for Enriching and Improving Textual Knowledge Bases》B Fetahu [Universität Hannover] (2018) O网页链接 view:O网页链接 ​​​​

《Learning Semantic Textual Similarity from Conversations》Y Yang, S Yuan, D Cer, S Kong, N Constant, P Pilar, H Ge, Y Sung, B Strope, R Kurzweil [Google] (2018) O网页链接 view:O网页链接 
    专利
    对于相同或相似的客服回复，其用户回复应该是相似的

《Sentence Simplification with Memory-Augmented Neural Networks》T Vu, B Hu, T Munkhdalai, H Yu [University of Massachusetts Amherst & University of Massachusetts Medical School & Microsoft Research] (2018) O网页链接 view:O网页链接 ​​​​

时间：北京时间 10月28号星期六早10点到中午12点
地点：将门斗鱼直播：https://www.douyu.com/jiangmen
祝贺小伙伴将CIFAR10刷到了非常接近Kaggle Leaderboard的第一名。我们同样会对这个星期里积极参与的同学赠予AWS Credit奖励。
上周大家纷纷对Aston的讲课激情点赞，这周Aston会继续给大家带来优化算法的高级部分。之后我们开始讲计算机视觉。
具体教程链接会在https://discuss.gluon.ai/t/topic/2080稍后更新。

学界｜一周论文：基于用户信息的神经网络对话模型
    http://www.weixinduba.com/n/164808
微软IJCAI2016演讲PPT：深度学习在语义理解上不再难有用武之地
    http://weibo.com/ttarticle/p/show?id=2309351000223996365343600455&u=2427818294&m=3996702359399303&cu=2427818294&ru=1402400261&rm=3996505783892101
【迎接人工智能时代，云创大数据发布DeepRack深度学习一体机】深度学习是一个效果很好但门槛极高的方向，如何落地产生实际应用效果成为关注的焦点。对此，厚积薄发的云创大数据(www.cstor.cn)打造了全新的深度学习软硬件平台，于2016年7月11日正式发布DeepRack深度学习一体机。
    http://www.cstor.cn/textdetail_10777.html
推荐一个Qiang Yang老师的大作，在多模态的背景下进行相似物体的搜索。如对一段文字搜索最相似的图片，或者对于一张图片搜索最相似的文字。论文提出一个大规模的异构转译哈希（HTH）算法。用哈希将数据映射到海明空间，再用转译函数对齐不同海明空间中的数据。@唐杰THU
    文字和图片的关联

https://github.com/TensorFlowKR
    awesome_tensorflow_implementations
        Sequence to Sequence -- chatbot
        Show and Tell: A Neural Image Caption Generator
    Pycon2016_ML(DL)
        list of Pycon2016 session related with ML
前辈之路(6) 张俊林专访 http://www.52cs.org/?p=1042
深度 | 微软人工智能首席科学家邓力：深度强化学习如何助力聊天机器人
    https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650717728&idx=1&sn=10d6dec16b2c2f7d569ea2ae206bbc54&scene=1&srcid=0803ULDtNEpLhAxY6B9xR7g8&pass_ticket=96aeuUUR7cpo3AfXrpcVFnO7ZL5zswPyXzN%2FtqR1fy21VVIRYm9V9Htr6DX6jh8V#rd
    对话成为移动 UI 的新兴范式
    作为智能对话接口代理的 bot
深度强化学习（Deep reinforcement learning，RL）
使用深度学习打造智能聊天机器人
    http://blog.csdn.net/malefactor/article/details/51901115
天津大学深度学习一线实战研讨班干货总结与资源下载
    https://mp.weixin.qq.com/s?__biz=MzI1NTE4NTUwOQ==&mid=2650325267&idx=1&sn=e89cfce72e1af9c32bd8be2440a9cfba&scene=1&srcid=0803xp3NxdlYxLe9ZvwthGf0&pass_ticket=96aeuUUR7cpo3AfXrpcVFnO7ZL5zswPyXzN%2FtqR1fy21VVIRYm9V9Htr6DX6jh8V#rd
    http://datasci.tju.edu.cn/data/index1?sukey=3997c0719f1515200399a26940a285f019a686a850fcc3d81290e00ce57e15e915fbabfbca74f113889c6a7bc0ce4a23

四位大咖为你解读人工智能的方方面面
    https://mp.weixin.qq.com/s?__biz=MzI3NjM4MjkyOA==&mid=2247483767&idx=1&sn=1c1775f772b7a67ccc2ba827f0141793&scene=0&pass_ticket=A5Jc%2BJJL9dX1zRXpYr6XolsBOI7LCKGjk8yaP1j8m8ZUlnHMWIGMj2%2Fv6zQkJ3W0#rd

TensorFlow tutorial
    https://github.com/martinwicke/TensorFlow-tutorial

专访 | 头条实验室科学家李磊：准确率更高的问答系统和概率程序语言
    介绍了也算是百度IDL在simple question上的KB-QA吧
    simple question就是只有一个relation的
    这里有启发的是对于实体的裁剪和关联（其实和句式中的type）很类似
    然后是先确定relation，而非先确定subject，这个从数量出发的考虑对于训练样本的充裕性也是有作用的
    只是看不少细节还很粗糙，我们NER+EL如何关联，relation和entity embedding的训练，特别是relation这块如何根据相关句子的embedding来得到relation之间的关系，也就是P(r|q)这个是比较有意义的，而P(s,r|q)的分解为两部分也可以借鉴，当然用我们自己的来做即可满足
    stacked BiGRU其实我觉得有点扯，当然对于问句做embedding的时候博瑜也可以看看
    CFO: Conditional Focused Neural Question Answering with Large-scale Knowledge Bases
    
    目前我们需要先rush出一个版本
    之后在框架和平台上，我也鼓励大家都尝试学习新的方法
    这样竞争力才够强
    能用数据解决的问题一定不要指望用算法
    但是不能全都是遇到问题的模板配置
    百度知道的数据先给张超这里
    张超你这里要用一些例子先将整个流程过一遍，可以在文档中体现
    虽然这会花费一点时间
    但是设计好，想清楚再做，其实比开始但发现要返工要来得高效
    这里我关心的包括属性或关系关联的问法，不同的分类，事件相关，你定义的pattern的各种分布
    估计instance数量比较少
    这个过程从明天开始做吧
    这周要完成
    每天要基于你写的进行讨论


Deep Learning Tutorial 286页 学习
    Lecture I: Introduction of Deep Learning
        Speech Recognition
        Handwritten Recognition
        Playing Go
        Dialogue System
        Network Structure -> Learning Target -> Learn!
        tensorflow torch theano libdnn caffe cntk chainer mxnet
    Lecture II: Tips for Training Deep Neural Network
    Lecture III: Variants of Neural Network
    Lecture IV: Next Wave

（2）百度知道数据
    获得百度知道问句总和是18450235句
    获得百度知道问句和相似问句总和是364858205句
（3）小白用户数据 200万条
    主要使用的是小白用户数据
    小白用户数据 主要问题是比较偏向小白功能，即句式不全面
    主要分析收集了该数据的句式，主要是根据动词来划分
（3）上周 最重要的就是更改了技术路线

    人物
        歌手
        演员
        普通人
    由于一个句子中的structure主要是属性和动词，所以今天手动根据属性和动词来寻找。

《Exploring Asymmetric Encoder-Decoder Structure for Context-based Sentence Representation Learning》S Tang, H Jin, C Fang, Z Wang, V R. d Sa [UC San Diego & Adobe Research] (2017) O网页链接 ​​​​   
《One-shot and few-shot learning of word embeddings》A K. Lampinen, J L. McClelland [Stanford University] (2017) O网页链接 ​​​​

Neural Sentiment Classification with User and Product Attention
    https://github.com/thunlp/NSC

PaddlePaddle
    https://github.com/baidu/paddle

Facebook
    https://github.com/facebook?comefrom=http://blogread.cn/news/

Facebook 开源了问答系统 DrQA，自动分析维基百科，回答用户的各种问题。O网页链接 ​​​​
    Reading Wikipedia to Answer Open-Domain Questions
    https://github.com/facebookresearch/DrQA
    
    https://github.com/facebookresearch 重要

Bengio等人提出MILABOT：强化学习聊天机器人
    


Percy Liang做了个很有意思的实验，就是通过给在阅读理解数据集中引入对抗样本。结果所有的模型都跪了。我们组胡明昊提出的Mnemonic Reader在对抗样本上远远好于其它模型。 ​​​​

推荐 苹果公司的机器学习博客
    https://machinelearning.apple.com/2017/07/07/GAN.html
基于对抗学习的生成式对话模型的坚实第一步 —— 始于直观思维的曲折探索
[DNN] 尝试理解深度神经网络的Large-batch魔咒
“GANs 之父”Goodfellow亲身传授：深度学习未来的8大方向和入门AI必备的三大技能
DeepLearningBook
    Ian Goodfellow, Yoshua Bengio and Aaron Courville
    MIT Deep Learning Book in PDF format
    https://github.com/HFTrader/DeepLearningBook

NIPS 2016 Tutorial: Generative Adversarial Networks

独家 | 阿里盖坤演讲：从人工特征到深度学习，我们为了更准确地预估点击率都做了多少努力 ( 附PPT )
    
    “TensorFlow 1.3.0-rc0 Released” O网页链接 ​​​​
    【AI背后的数学原理之RNN】《Recurrent Neural Networks - The Math of Intelligence (Week 5) - YouTube》by Siraj Raval  O网页链接 GitHub:O网页链接 ​​​​

刘知远
    Chinese NLP
    THULAC: An Efficient Lexical Analyzer for Chinese. [homepage][Git C++][Git Java][Git Python]
    THUCTC: An Efficient Chinese Text Classifier. [homepage][Git Java]

    https://github.com/thunlp/NRE
    http://thuctc.thunlp.org/

Supervised Sequence Labelling with Recurrent Neural Networks 已看

NIPS 2016 Tutorial: Generative Adversarial Networks
    
Keras中文文档

Deep Learning Book
    Ian Goodfellow, Yoshua Bengio and Aaron Courville
    
    Eigendecomposition
    1.2 Historical Trends in Deep Learning 20170720
    
    一 1
    二 31 7.27 7.28
    三 53
    四 80
    五 98
    六 167
    七 228
    八 274
    九 331
    十 374
    十一 424
    十二 446
    十三 489
    十四 505
    十五 529
    十六 561
    十七 593
    十八 608
    十九 634
    二十 656
    
A Primer on Neural Network Models for Natural Language Processing



AI100 头条 http://top.ai100.ai/

机器之心 http://www.jiqizhixin.com/
    账号 zhch_lrgy@163.com PIN+
    自然语言处理领域深度学习研究总结：从基本概念到前沿成果
    
PaperWeekly http://rsarxiv.github.io/

新智元

智能立方

爱老师爱可可

好东西传送门

AI科技评论

深度学习大讲堂

http://dataunion.org/

http://www.machinedlearnings.com/

腾讯研究院
    http://www.tisi.org/m54

http://www.infoq.com/cn/
    
Chatbots技术与产品
    总结│解密 chatbot 人工智能聊天机器人 技术沙龙

2017年1月
    2017年1月1日
    【推荐】Pandas数据处置速查表
    【学习】决策树在商品购买能力预测案例中的算法实现
    【学习】变の贝叶斯
    【学习】Intel 收官开源之作--BigDL：构建在 Apache Spark 之上的分布式深度学习库
    2017年1月2日
    【推荐】阿里天池O2O优惠券消费行为预测竞赛优胜方案
    【学习】tensorflow笔记：多层LSTM代码分析
    【学习】机器学习基础训练营(1.23–1.27, 讲师来自UC Berkeley, CMU等)
    【学习】Theano tutorial和卷积神经网络的Theano实现 Part1
    2017年1月3日
    【推荐】Ian Goodfellow的NIPS 2016教学讲座: GANs现已写成介绍性论文
    【学习】加大伯克利分校计算机课程CS 294，2017年春季，深度强化学习
    【学习】(论文+代码)Feedback Networks —— 反馈网络
    【报名】微软亚洲研究院王太峰：浅谈分布式机器学习算法和工具
    2017年1月4日
    【推荐】用Spark分析Amazon八千万商品评价
    【数据集】时间序列/信号处理/医学影像开源数据集
    【学习】使用Python的时间序列数据可视化
    【学习】百度PaddlePaddle深度学习平台：面向工程师，性能优先
    2017年1月5日
    【推荐】NLP必读经典文献100篇
    【干货】循环神经网络（Recurrent）——介绍
    【学习】基于Spark GraphX实现微博二度关系推荐实践
    【学习】机器码农：深度学习自动编程
    2017年1月6日
    【推荐】手把手：TensorFlow实现简单图片识别系统
    【干货】微软亚洲研究院王太峰：浅谈分布式机器学习算法和工具
    【学习】直观理解信息论
    【论文】Neural style transfer的理论解释
    2017年1月7日
    【推荐】神经网络与深度学习
    【干货】深度学习新星：GANs的基本原理、应用和走向
    【论文】(论文+代码)苹果发表的AI方面第一篇论文
    【课程】大数据科学与应用系列讲座（自主模式）
    2017年1月8日
    【推荐】AlphaGo核心部分的Python原生重现
    【学习】Bring TensorBoard to MXNet
    【论文】Deep Reinforcement Learning for Dialogue Generation
    【学习】技术人，为什么需要构建知识图谱
    2017年1月9日
    【推荐】机器学习法则：机器学习工程最佳实践
    【学习】(论文+代码+数据)SalGAN：用生成对抗性网络进行视觉显著性预测
    【论文】Deep Convolutional Denoising of Low-Light Images
    【学习】350+ 数据结构编程面试问题
    2017年1月10日
    【推荐】NIPS 2016 Tutorial: 变分推断
    【学习】TensorFlow + Inception + Raspberry Pi的视频连续分类(用CNN检测视频内容)
    【学习】没有任何公式——直观的理解变分自动编码器VAE
    【学习】增强学习的解释——学习基于长期回报的行为
    2017年1月11日
    【推荐】关于机器学习的领悟与反思 —— 张志华
    【课程】麻省理工学院（MIT）公开课.S094：自主驾驶汽车的深度学习
    【学习】这些杀手级应用不太冷——从语义网到知识图谱的回顾
    【报名】微软亚洲研究院边江：机器学习驱动下的内容分发和个性化推荐
    2017年1月12日
    【推荐】THUOCL：清华大学开放中文词库
    【学习】基于Keras实现CNN验证码识别的知乎爬虫
    【学习】Cross-Validation（交叉验证）详解
    【学习】对深度网络Dropout层的详尽分析
    2017年1月13日
    【推荐】Python自然语言处理(NLP)实践指南
    【干货】微软亚洲研究院边江：机器学习驱动下的内容分发和个性化推荐
    【学习】通俗理解神经网络BP传播算法
    【学习】美团Apache Kylin精确去重指标优化历程
    2017年1月14日
    【推荐】Data Science Bowl 2017：肺癌检测竞赛(60多GB数据&奖金1百万美金)
    【干货】2016CCF 大数据精准营销中搜狗用户画像挖掘优胜方案
    【学习】基于深度学习的交通灯检测器，使用dlib和几个来自谷歌街景的图像
    【学习】Integrating Lexical Contrast into Word Embeddings
    2017年1月15日
    【推荐】用深度学习识别红绿灯，介绍作者如何在10周内学习深度学习，并赢得了5000美元奖金
    【论文】《Machine Learning that Matters》 论文辩证讨论ML研究的重要性和必要性（不要仅灌水）
    【学习】推荐系统中基于深度学习的混合协同过滤模型
    【学习】优化常用不等式速查
    2017年1月16日
    【抢课】人工智能前沿与产业趋势——与10余位顶级大咖面对面
    【学习】清华大学刘知远老师为本科生入门自然语言处理做的推荐书目
    【论文】Visualizing Residual Networks
    【报名】MIT在读博士周博磊：理解和利用CNN的内部表征
    2017年1月17日
    【推荐】通过LSTM/CNN对抗训练生成文本
    【学习】机器学习算法线上部署方法
    【学习】金融大数据在平安科技信用风险管理中的应用实践
    【抢课】人工智能前沿与产业趋势——与10余位顶级大咖面对面
    2017年1月18日
    【推荐】MIT在读博士周博磊：理解和利用CNN的内部表征
    【学习】《不一样的技术创新-阿里巴巴双11背后的技术》PDF开放下载！
    【论文】理解深度卷积网络的有效感受野(ERF)
    【学习】可能是 2017 最全的机器学习开源项目列表
    2017年1月19日
    【推荐】Python数据科学之NumPy速查
    【学习】一篇很好的参考文章：深度学习算法在自然语言处理中的一些心得
    【学习】 饿了么推荐系统：从0到1
    【学习】数据挖掘 知识重点（整理版）
    2017年1月20日
    【推荐】对抗生成网络(GAN)训练过程研究
    【学习】深度学习论文实现：空间变换网络
    【干货】DBoW3 视觉词袋模型、视觉字典和图像数据库分析
    【学习】CIPS青工委学术专栏第19期 | 基于互指导的实体之间相关度计算方法
    2017年1月21日
    【推荐】自然语言处理NLP推荐学习路线及参考资料
    【学习】（教程+代码）基于Keras＆Tensorflow的车道跟随自动驾驶仪
    【学习】 2017年最值得关注的科学概念之“迁移学习”
    【学习】Sequential Match Network
    2017年1月22日
    【推荐】不用博士学位玩转Tensorflow深度学习
    【学习】跨领域推荐，实现个性化服务的技术途径
    【学习】基于 Gensim 的 Word2Vec 实践
    【学习】技术学习年货之--交易核心链路的故事
    2017年1月23日
    【推荐】(R/Python)t-SNE聚类算法实践指南
    【论文】（论文+代码）无监督的跨域图像生成
    【学习】汉字生成模型的那些坑
    【学习】深度增强学习系列文章
    2017年1月24日
    【推荐】Kaggle大牛讲解Gradient Boosting基础
    【学习】 文本型医疗大数据，拿来就可用？
    【学习】深度学习第一课：了解深度学习的基本原理和工作方式
    【学习】从零开始的Python爬虫速成指南
    2017年1月25日
    【推荐】基于CGAN的图像除雨方法(ID-CGAN)
    【学习】 如何做出一个更好的Machine Learning预测模型
    【学习】如何用Python和PubNub做实时数据可视化
    【学习】Intel开源深度学习库BigDL：Non GPU on Spark
    2017年1月26日
    【推荐】Apache Spark/Keras分布式深度学习
    【学习】用Spark和DBSCAN对地理定位数据进行聚类
    【学习】深度神经网络进军医学界——《自然》杂志刊登了一篇斯坦福大学关于用深度学习检测皮肤癌的论文
    【学习】线性相关和秩的物理意义
    2017年1月27日
    【推荐】深度增强学习概览
    【学习】深度学习：今生前世 —— 张志华
    【学习】理解NLP深度学习模型
    【学习】这一年来，数据科学家都用哪些算法？
    2017年1月28日
    【推荐】机器学习常用公式集
    【学习】部署机器学习简明指南
    【学习】NLP/深度学习/增强学习/AI主题Python代码集锦
    【学习】用LaTeX绘制贝叶斯网络、图模型、(有向)因子图等
    2017年1月29日
    【推荐】Google 研究员、Keras 作者Francois Chollet书籍《Python深度学习》
    【学习】知名机器学习专家Russ Salakhutdinov的深度学习教程
    【推荐】基于数据分析的最佳数据科学在线课程推荐
    【论文】通过对齐学习实现端到端的人脸识别
    2017年1月30日
    【推荐】考察数据科学家深度学习基础的45道题(及答案)
    【学习】真的理解贝叶斯公式吗？
    【学习】编程基础书籍自由电子版新春大放送
    【学习】赛尔译文 | Dropout分析
    2017年1月31日
    【推荐】AI突破性论文及代码实现汇总
    【学习】增强学习能解决什么样的机器人问题？CMU深度增强学习第一讲
    【学习】OpenCV(C++/Python)手写数字分类教程
    【学习】概率编程核心算法
2017年2月
    2017年2月1日
    【推荐】利用深度学习的人脸检测方法：改进的Faster RCNN
    【学习】Pandas 秘籍
    【学习】利用python爬取人人贷网的数据
    【学习】人工智能自学心得
    2017年2月2日
    【推荐】用遗传算法选择(迁移学习)梯度下降通道的大型网络PathNet
    【学习】又一个生活中的贝叶斯应用
    【学习】利用Tensorflow从一种图像转译成另一种图像的生成技术
    【学习】天文学家利用GAN生成图像改进识别系统
    2017年2月3日
    【推荐】机器学习速查：算法选择指南
    【学习】（论文+代码）图像内容更改的深层特征插值
    【论文】根据特定角度的特征选择性指数索引神经元，进而实现CNN模型可视化
    【学习】 免费电子书《认知的概率模型》下载+代码
    2017年2月4日
    【推荐】机器学习资源汇总：Kaggle方案/教程/讲义/链接
    【论文】谷歌大脑团队论文：Pixel Recursive Super Resolution
    【学习】NanoNets :数据有限如何应用深度学习
    【学习】Google官博解读：如何用机器学习解决停车难题
    2017年2月5日
    【推荐】Python机器学习：Scikit-Learn教程
    【学习】详解反向传播算法(上)
    【论文】FlowNet 2.0：利用深度网络进行光流估计的演进
    【学习】当AI邂逅艺术：机器写诗综述
    2017年2月6日
    【推荐】AAAI 2017的Tutorial——讲述了深度学习框架的设计思想和实现，比较若干种流行框架的性能和异同
    【学习】考察数据科学家聚类技术的40道题(及答案)
    【学习】纯干货：深度学习实现之空间变换网络-part1
    【学习】小米品牌广告引擎与算法实践
    2017年2月7日
    【推荐】（论文+代码+数据）AAAI 2017论文：基于深度学习的城市人流预测
    【学习】不容错过的2016年机器学习10大文章（v.2017）
    【学习】中国人工智能学会通讯2017年第一期
    【学习】最轻松的学习Tensorflow技术
    2017年2月8日
    【推荐】CNN和RNN在自然语言处理中的比较研究
    【学习】牛津大学2017学年自然语言处理高级课程-
    【论文】实时目标检测的Wide-Residual-Inception网络
    【学习】用Spark训练大规模语言模型
    2017年2月9日
    【推荐】人脸老化GAN模型(acGAN)，相比其他方法，可更好地保持原图面部特征
    【学习】迁移学习：数据不足时如何深度学习
    【学习】Tensorflow机器学习宝典的代码
    【学习】MDP中的规划-CMU深度强化学习第三讲
    2017年2月10日
    【推荐】百度深度学习平台PaddlePaddle的深度学习入门教程
    【干货】东南大学教授漆桂林：知识图谱中的推理技术进展及应用
    【学习】深度学习在美团点评的应用
    【论文】（论文+代码）Pixel Objectness —— 更好的自动抠图、图像检索、图像重定向技术
    2017年2月11日
    【推荐】人工智能免费入门课程
    【学习】CTR预估中的贝叶斯平滑方法及其代码实现
    【学习】初学者可以在几分钟内构建的6种深度学习应用程序（使用Python）
    【干货】为什么梯度反方向是函数值下降最快的方向？
    2017年2月12日
    【推荐】效果出乎意料的1x1卷积
    【学习】八一八聊天机器人
    【学习】(PyTorch教程+代码)50行代码实现对抗生成网络(GAN)
    【干货】软件工程在谷歌
    2017年2月13日
    【推荐】微软亚洲研究院的刘铁岩等人在AAAI 2017上做的有关优化以及大规模机器学习的Tutorial
    【数据集】人工智能领域比较常见的数据集汇总
    【学习】用Spark机器学习数据流水线进行广告检测
    【报名】MIT在读博士生吴佳俊：生成和识别三维物体
    2017年2月14日
    【推荐】来自谷歌Batch Normalization原作者的论文Batch Renormalization
    【学习】考察数据科学家集成方法的40道题(及答案)
    【学习】基于TensorFlow让机器生成赵雷曲风的歌词
    【报名】今日头条科学家、头条实验室总监李磊：大规模知识库上的自然语言问答
    2017年2月15日
    【推荐】条条大路通罗马LS-GAN：把GAN建立在Lipschitz密度上
    【学习】用ArtGAN合成现实艺术作品
    【学习】使用OpenCV的目标跟踪技术（C ++ / Python）
    【学习】机器学习中，有哪些特征选择的工程方法？
    2017年2月16日
    【推荐】MIT在读博士生吴佳俊：生成和识别三维物体
    【论文】（论文+代码）软权值共享的神经网络压缩技术
    【学习】淘宝搜索/推荐系统背后深度强化学习与自适应在线学习的实践之路
    【讲座】深度学习算法、框架和性能优化介绍
    2017年2月17日
    【推荐】不可思议的PyTorch：PyTorch教程、论文、项目、社区资源大汇总
    【论文】随机配置网络：原理与算法
    【干货】今日头条科学家、头条实验室总监李磊：大规模知识库上的自然语言问答
    【学习】未来已来！阿里小蜜AI技术揭秘
    2017年2月18日
    北大AI公开课：徐小平、雷鸣 -- 人工智能的发展、挑战与机遇
    【学习】利用TensorFlow搞定知乎验证码之《让你找中文倒转汉字》
    【学习】形象的解释神经网络激活函数的作用是什么？
    【讲座】研发面向健康的AlphaGo
    2017年2月19日
    【推荐】思维导图：机器学习算法分类速查
    【学习】浅析生成对抗网络
    【学习】贝叶斯原理及其推断简介
    【论文】(论文+Caffe代码+数据集)时尚元素标定
    2017年2月21日
    【推荐】深度学习在医学图像分析上应用的综述（34页）
    “魔镜杯”互联网金融数据训练营报名启动！仅100个名额！
    【论文】深度随机配置网络:万局逼近与学习表示
    【学习】(Python/R)SVM及调参教程
    2017年2月22日
    徐小平对话雷鸣——AI 创业仅有科学家是万万不行的
    北大AI公开课第一讲：雷鸣评人工智能前沿与行业结合点（附PPT和最新高清回放链接）
    【干货】面向机器学习、数据科学、概率、SQL及大数据的28个最热门的速查表
    【论文】（论文+代码）加速的PixelCNN ++，图像生成效率提升了183倍
    【学习】机器学习若干问题的新颖，可证明，实用算法
    2017年2月23日
    【推荐】卷积神经网络中稀疏性的能力
    【学习】掌握 Google 深度学习框架的正确姿势——专访 TensorFlow 贡献者唐源
    【学习】机器学习系列-广义线性模型
    【学习】Facebook 机器学习@Scale 2017 资料汇总
    2017年2月24日
    【推荐】阿里开源强化学习研究平台Gym StarCraft
    【学习】迎接自然语言处理新时代
    【论文】WSDM 2017论文：如何把强化学习融合到广告中的Real-Time Bidding中
    将门高欣欣对话AI天团：人工智能的技术前瞻和商业的未来
    2017年2月25日
    北大AI公开课第二讲：余凯、雷鸣---嵌入式人工智能：从边缘开始的革命
    【学习】为什么Kaggle数据分析竞赛者偏爱XGBoost
    【学习】人人都可以做深度学习应用：入门篇
    【论文】特征学习顶会ICLR 2017三篇最佳论文
    2017年2月26日
    【推荐】一个非常棒的可视化概率及统计的学习网站
    【论文】WSDM 2017论文：Recurrent Recommender Networks
    【学习】使用传统的计算机视觉和机器学习技术为进行车辆跟踪和检测
    【学习】(Python)白手起家机器学习
    2017年2月27日
    北大AI公开课第二讲：余凯、雷鸣---嵌入式人工智能：从边缘开始的革命
    【干货】基础机器学习算法
    【学习】人脸识别中的活体检测
    【学习】推荐系统本质与网易严选实践
    2017年2月28日
    【推荐】用于目标检测相关的资源列表
    【学习】论深度学习的起源（81页+200参考文献）
    【论文】（论文+代码）基于级联全卷积神经网络和三维条件随机场的肝脏与病变自动分割
    【学习】梯度下降与反向传播（上）
2017年3月
    2017年3月1日
    【推荐】从零开始学习无人驾驶技术 --- 车道检测
    【学习】Kaggle老手领你入门梯度提升——梯度提升两三事
    【论文】用对抗网络检测恶性前列腺癌
    【学习】Neural Relation Extraction（NRE）
    2017年3月2日
    北大 AI 公开课第二讲：余凯&雷鸣漫谈嵌入式AI（附课程精彩回放）
    【预告】北大AI公开课第三讲：漆远、雷鸣---人工智能驱动的金融生活服务
    【学习】七步进阶Python机器学习
    重磅丨硅谷人工智能公开课
    2017年3月3日
    【预告】北大AI公开课第三讲：漆远、雷鸣---人工智能驱动的金融生活服务
    【学习】职业转换：从量化金融到机器学习
    【学习】中国人工智能学会通讯2017年第二期
    北大 AI 公开课第二讲：余凯&雷鸣漫谈嵌入式AI（附课程精彩回放）
    2017年3月4日
    【推荐】李理：卷积神经网络之Batch Normalization的原理及实现
    【干货】2010-2017最全KDD CUP赛题回顾及数据集下载
    【学习】使用sklearn做特征工程
    【学习】FAQ聊天机器人构建指南：文本内容预测任务实例
    2017年3月5日
    【推荐】详解反向传播算法(下)
    【论文】（论文+代码）ShaResNet: 通过共享权值减少残差网络参数
    【学习】用机器学习判定红楼梦后40回是否为曹雪芹所写？
    【学习】使用sklearn进行数据挖掘
    
https://engineering.quora.com/
    Semantic Question Matching with Deep Learning
    Updates to Our Bug Bounty Program
    
A Persona-Based Neural Conversation Model
    We present persona-based models for handling the issue of speaker consistency in neural response generation. A speaker model encodes personas in distributed embeddings that capture individual characteristics such as background information and speaking style. A dyadic speaker-addressee model captures properties of interactions between two interlocutors. Our models yield qualitative performance improvements in both perplexity and BLEU scores over baseline sequence-to-sequence models, with similar gains in speaker consistency as measured by human judges.

系列:自己动手做聊天机器人 http://www.shareditor.com/blogshow/?blogId=63
    一-涉及知识
    二-初识NLTK库
    三-语料与词汇资源
    四-何须动手？完全自动化对语料做词性标注
    五-自然语言处理中的文本分类
    六-教你怎么从一句话里提取出十句话的信息
    七-文法分析还是基于特征好啊
    八-重温自然语言处理
    九-聊天机器人应该怎么做
    十-半个小时搞定词性标注与关键词提取
    十一-0字节存储海量语料资源
    十二-教你如何利用强大的中文语言技术平台做依存句法和语义依存分析
    十三-把语言模型探究到底
    十四-探究中文分词的艺术
    十五-一篇文章读懂拿了图灵奖和诺贝尔奖的概率图模型
    十六-大话自然语言处理中的囊中取物
    十七-让机器做词性自动标注的具体方法
    十八-神奇算法之句法分析树的生成
    十九-机器人是怎么理解“日后再说”的
    二十-语义角色标注的基本方法
    二十一-比TF-IDF更好的隐含语义索引模型是个什么鬼
    二十二-神奇算法之人工神经网络
    二十三-用CNN做深度学习
    二十四-将深度学习应用到NLP
    二十五-google的文本挖掘深度学习工具word2vec的实现原理
    二十六-图解递归神经网络(RNN)
    二十七-用深度学习来做自动问答的一般方法
    二十八-脑洞大开：基于美剧字幕的聊天语料库建设方案
    二十九-重磅：近1GB的三千万聊天语料供出
    三十-第一版聊天机器人诞生——吃了字幕长大的小二兔
    三十一-如何把网站流量导向小二兔机器人
    三十二-用三千万影视剧字幕语料库生成词向量
    三十三-两套代码详解LSTM-RNN——有记忆的神经网络
    三十四-最快的深度学习框架torch
    三十五-一个lstm单元让聊天机器人学会甄嬛体
    三十六-深入理解tensorflow的session和graph
    三十七-一张图了解tensorflow中的线性回归工作原理
    三十八-原来聊天机器人是这么做出来的
    三十九-满腔热血：在家里搭建一台GPU云服务共享给人工智能和大数据爱好者
    四十-视频教程之开篇宣言与知识点梳理
    
开发者课堂
    https://www.youtube.com/watch?v=kjhiXQfaFeo&list=PLO5e_-yXpYLARtW5NPHTFVYY-xpgwuNNH http://www.maiziedu.com/course/373-3811/
        1.课程介绍机器学习介绍上23:09                     1.18
        2.课程介绍机器学习介绍下04:32                    1.18
        3.深度学习介绍26:47                                1.18
        4.基本概念34:48
        5.决策树算法39:15
        6.决策树应用37:28
        7.最邻近规则分类KNN算法28:03
        8.最邻近规则KNN分类应用31:46
        9.支持向量机SVM上35:40
        10.支持向量机SVM上应用26:27
        11.神经网络算法应用上49:51
        12.神经网络算法应用下21:21
        13.简单线性回归上29:49
        14.简单线性回归下28:01
        15.多元线性回归33:53
        16.多元线性回归应用29:46
        17.非线性回归 Logistic Regression32:34
        18.非线性回归应用29:25
        19.神经网络NN算法56:16
        20.支持向量机(SVM)算法(下)应用29:55
        21.支持向量机(SVM)算法下25:08
        22.回归中的相关度和决定系数32:24
        23.回归中的相关性和R平方值应用24:00
        24.Kmeans算法33:20
        25.Kmeans应用36:06
        26.Hierarchical clustering 层次聚类19:15

    https://www.youtube.com/watch?v=xy7KlALq2n8&list=PLO5e_-yXpYLDyeADG7xbFaJxvYyeAjHah
        1.基本概念清晰版27:03                            1.18
            AI > Machine Learning > Representation Learning > Deep Learning
        2.软件包安装和环境配置总述33:16
        3.环境配置分部详解32:27
        4.环境配置分部详解下35:07
        5.手写数字识别28:08
        6.神经网络基本结构及梯度下降算法43:01
        7.随机梯度下降算法12:13
        8.梯度下降算法实现上26:05
        9.梯度下降算法实现下32:27
        10.神经网络手写数字演示40:14
        11.Backpropagation算法上33:39
        12.Backpropagation算法下33:11
        13.Backpropagation算法实现30:46
        14.cross-entropy函数27:06
        15.Softmax和Overfitting39:55
        16.Regulization21:45
        17.Regulazition和Dropout31:19
        18.正态分布和初始化(修正版)17:08
        19.提高版本的手写数字识别实现30:47
        20.神经网络参数hyper-parameters选择31:07
        21.深度神经网络中的难点36:56
        22.用ReL解决VanishingGradient问题16:38
        23.ConvolutionNerualNetwork算法29:40
        24.ConvolutionNeuralNetwork实现上30:50
        25.ConvolutionNeuralNetwork实现下22:16
        26.Restricted Boltzmann Machine24:53
        27.Restricted Boltzmann Machine下20:31
        28.Deep Brief Network 和 Autoencoder19:29

Tensorflow
    Tensorflow tutorials (Eng Sub) 神经网络 教学 教程 
        https://github.com/MorvanZhou
        https://github.com/MorvanZhou/tutorials
        教程链接 https://www.youtube.com/playlist?list=PLXO45tsB95cKI5AIlf5TxxFPzb-0zeVZ8
        代码链接 https://github.com/MorvanZhou/tutorials/tree/master/tensorflowTUT
        什么是神经网络 (机器学习) what is neural network in machine learning                                1.20
        Tensorflow 1 why? (神经网络 教学教程tutorial)                                                        1.20
        Tensorflow 2 安装 (Windows, Mac, Linux) (神经网络 教学教程tutorial)                                    1.20
        Tensorflow 3 例子1 (神经网络 教学教程tutorial)                                                        1.20
        Tensorflow 4 处理结构 (神经网络 教学教程tutorial)                                                    1.20
        Tensorflow 5 例子2 (神经网络 教学教程tutorial)                                                        1.20
        Tensorflow 6 Session 会话控制 (神经网络 教学教程tutorial)                                            1.20
        Tensorflow 7 Variable 变量 (神经网络 教学教程tutorial)                                                1.20
        Tensorflow 8 placeholder 传入值 (神经网络 教学教程tutorial)                                            1.20
        机器学习技巧4: 为什么需要激励函数 (深度学习)? Why need activation functions (deep learning)?        1.20
        Tensorflow 9 激励函数 activation function (神经网络 教学教程tutorial)                                1.20
        Tensorflow 10 例子3 添加层 def add_layer() (神经网络 教学教程tutorial)                                1.20
        Tensorflow 11 例子3 建造神经网络 build a neural network (神经网络 教学教程tutorial)                    1.20
        Tensorflow 12 例子3 结果可视化 plot result (神经网络 教学教程tutorial)                                1.20
        机器学习技巧6: 加速神经网络训练 (深度学习) Speed up neural network training process (deep learning)    1.20
        Tensorflow 13 优化器 optimizer (神经网络 教学教程tutorial)                                            1.22
        Tensorflow 14 Tensorboard 可视化好帮手 (神经网络 教学教程tutorial)                                    1.22
        Tensorflow 15 Tensorboard 可视化好帮手2 (神经网络 教学教程tutorial)                                    1.22
        Tensorflow 16 Classification 分类学习 (神经网络 教学教程tutorial)                                    1.22
        机器学习技巧5: 什么是过拟合 (深度学习)? What is overfitting (deep learning)?                        1.22
        Tensorflow 17 dropout 解决 overfitting 问题 (神经网络 教学教程tutorial)                                1.22
        #3 什么是卷积神经网络 CNN (深度学习)? What is Convolutional Neural Networks (deep learning)?        1.22
        Tensorflow 18.1 CNN 卷积神经网络 Convolutional Neural Networks 1 (神经网络 教学教程tutorial)        1.22
        Tensorflow 18.2 CNN 卷积神经网络 Convolutional Neural Networks (神经网络 教学教程tutorial)            1.22
        Tensorflow 18.3 CNN 卷积神经网络 Convolutional Neural Networks (神经网络 教学教程tutorial)            1.22
        Tensorflow 19 Saver 保存读取 (神经网络 教学教程tutorial)                                            1.22
        #4 什么是循环神经网络 RNN (深度学习)? What is Recurrent Neural Networks (deep learning)?            1.23
        Tensorflow 20.1 RNN 循环神经网络 (神经网络 教学教程tutorial)                                        1.23
        Tensorflow 20.2 RNN lstm 循环神经网络 (分类例子) (神经网络 教学教程tutorial)                        1.23
        Tensorflow 20.3 RNN lstm (regression 回归例子) (神经网络 教学教程tutorial)                            1.23
        #5 什么是 LSTM RNN 循环神经网络 (深度学习)? What is LSTM in RNN (deep learning)?                    1.23
        Tensorflow 20.4 RNN lstm (回归例子可视化) (神经网络 教学教程tutorial)                                1.23
        #6 什么是自编码 Autoencoder (深度学习)? What is an Autoencoder in Neural Networks (deep learning)?    1.23
        Tensorflow 21 自编码 Autoencoder (非监督学习) (神经网络 教学教程tutorial)                            
        Tensorflow 22 name_scope/ variable_scope 命名方式 (神经网络 教学教程 tutorial)                        
        机器学习技巧8: 为什么要 Batch Normalization 批标准化 (深度学习 deep learning)                        
        Tensorflow 23 Batch normalization 批标准化 (神经网络 教学教程tutorial)                                

Factorization Machines 学习笔记
    http://blog.csdn.net/itplus/article/details/40534885
    Factorization Machines 学习笔记（一）预测任务
    Factorization Machines 学习笔记（二）模型方程
    Factorization Machines 学习笔记（三）回归和分类
    Factorization Machines 学习笔记（四）学习算法

无痛的机器学习第一季目录 https://zhuanlan.zhihu.com/p/22464594?refer=hsmyy
     冯超 · 3 个月前
    经过5个月的努力，我终于完成了40篇不高不低还算有些干货的机器学习文章。回首看看这5个月的努力，每一次的写作都充满了开心与痛苦。说开心是因为当自己完成每一个章节的写作后，自己感觉对这一部分的知识有了更加深刻地认识，而痛苦则是对写作过程中一系列事情的恐惧——找不到好选题，对论文细节的困惑，跑不出想要的结果，难以用通俗易懂的语言描述自己所知……好在这一切就要告一段落了。
    以下就做一个集合贴，展示一下这一季的所有文章，对本专栏文章感兴趣的童鞋，收藏这一篇就足够了（后续未发布的几篇和番外篇会更新上来）：
    文章目录
    CNN网络基础结构
        神经网络-全连接层（1）
        神经网络-全连接层（2）
        神经网络-全连接层（3）
        卷积层（1）
        卷积层（2） 
        卷积层（3）
    CNN网络上层结构
        CNN——架构上的一些数字
        CNN--结构上的思考
        CNN Dropout的极端实验
    Caffe源码分析
        Caffe代码阅读——层次结构 
        Caffe源码阅读——Net组装 
        Caffe代码阅读——Solver 
        CNN--两个Loss层计算的数值问题 
        Caffe源码阅读——DataLayer&Data Transformer
    生成网络
        DCGAN的小尝试（1）
        DCGAN的小尝试（2）
        VAE（1）——从KL说起
        VAE(2)——基本思想
        VAE(3)——公式与实现
        VAE（4）——实现
    优化算法
        梯度下降是门手艺活…… 
        路遥知马力——Momentum 
        CNN——L1正则的稀疏性
        Caffe中的SGD的变种优化算法(1)
        Caffe中的SGD的变种优化算法(2)
    CNN可视化
        CNN-反卷积（1）
        CNN-卷积反卷积（2）
        寻找CNN的弱点
    CNN数值
        CNN的数值实验 - 无痛的机器学习 - 知乎专栏
        CNN数值——xavier（上） - 无痛的机器学习 - 知乎专栏
        CNN数值——xavier（下） - 无痛的机器学习 - 知乎专栏
        CNN数值——ZCA - 无痛的机器学习 - 知乎专栏
    FCN
        FCN(1)——从分类问题出发
        FCN(2)——CRF通俗非严谨的入门
        FCN(3)——DenseCRF
        FCN(4)——Mean Field Variational Inference
        FCN(5)——DenseCRF推导
        FCN(6)——从CRF到RNN
    Representation
        CenterLoss——实战&源码
    GPU
        [翻译]Exploring the Complexities of PCIe Connectivity and Peer-to-Peer Communication - 无痛的机器学习 - 知乎专栏
    “聊点轻松的”系列
        聊点轻松的——划个水
        聊点轻松的2——斗图篇
        聊点轻松的3——什么是学习
        聊点轻松的4——这回真的很轻松
        聊点轻松的5——这篇写得并不轻松
    番外篇
        番外篇(1)——最速下降法
        番外篇(2)——无聊的最速下降法推导
        番外篇(3)——最速下降法的特点
        番外篇(4)——共轭梯度法入坑
        番外篇(5)——共轭方向的构建
        番外篇(6)——共轭梯度的效果
    
LDA
    LDA漫游指南
        http://yuedu.baidu.com/ebook/d0b441a8ccbff121dd36839a
        封面简介 
        作品简介
        作者简介
        前言 
        第1章 背景 
        第2章 前置知识
            beta函数 0-1 曲线面积
        第3章 LDA的Gibbs Sampling推导
        第4章 实现与应用
        第5章 并行化
        第6章 变分贝叶斯的启蒙
        第7章 LDA的变分贝叶斯法
        第8章 LDA变分EM实现
        附录
    用机器学习研究UFO目击报告！数据科学之魅：隐含狄利克雷分布
        https://yq.aliyun.com/articles/68521?spm=5176.8279002.620388.8
    
数学
    怎么来理解伽玛（gamma）分布？    https://www.zhihu.com/question/34866983
    怎么通俗易懂地解释贝叶斯网络和它的应用？    https://www.zhihu.com/question/28006799
    
白话大数据与机器学习
    共18章
    22号 1-4章
    23号 5-7章
    24号 8-9章
    25号 10-11章
    26号 12-13章
    27号 14-15章
    28号 16-17章
    29号 18章
    
自然语言处理知识列表
    MIT自然语言处理 (23)
    PRML (15)
    Topic Model (10)
    wordpress (6)
    专题 (6)
    中文信息处理 (23)
    中文分词 (40)
    并行算法 (1)
    招聘 (4)
    推荐系统 (3)
    数据挖掘 (2)
    文本分类 (3)
    文本处理演示系统 (3)
    最大熵模型 (7)
    机器学习 (33)
    机器翻译 (54)
    条件随机场 (3)
    标注 (15)
    深度学习 (4)
    科学计算 (1)
    统计学 (10)
    翻译模型 (2)
    自然语言处理 (237)
    计算语言学 (39)
    词典 (8)
    语义学 (1)
    语义相似度 (1)
    语义网 (3)
    语料库 (12)
    语言模型 (24)
    语音识别 (4)
    贝叶斯模型 (1)
    转载 (28)
    问答系统 (1)
    随笔 (63)
    隐马尔科夫模型 (37)
    
计算机编程
    正则表达式
        http://mp.weixin.qq.com/s?__biz=MjM5MDI5MjAyMA==&mid=2651384401&idx=1&sn=26d1c3b8fde6e836a21ef4ead28c6b34&chksm=bdbb4b0a8accc21ca7cf978170da5a52c9a86843890211a3f1b67f05b608b2e94e9fe4a2466d#rd
        这20个正则表达式，让你少写1,000行代码

黄锦池-hjimce http://my.csdn.net/hjimce
    机器学习基础实践
        机器学习（十四）Libsvm学习笔记
        机器学习（十三）k-svd字典学习
        机器学习（十二）朴素贝叶斯分类
        机器学习（十一）谱聚类算法
        机器学习（十）Mean Shift 聚类算法
        机器学习（九）初识BP神经网络
        机器学习（八）Apriori算法学习
        机器学习（七）白化whitening
        机器学习（六）非负矩阵分解NMF-未完待续
        机器学习（五）PCA数据降维
        机器学习（四）高斯混合模型
        机器学习（三）k均值聚类
        机器学习（二）逻辑回归
        机器学习（一）线性回归
    深度学习
        深度学习（四十二）word2vec词向量学习笔记
        深度学习（四十一）cuda8.0+ubuntu16.04+theano、caffe、tensorflow环境搭建
        深度学习（四十）caffe使用点滴记录
        深度学习（三十九）可视化理解卷积神经网络(2.0)
        深度学习（三十八）卷积神经网络入门学习(2.0)
        深度学习（三十七）优化求解系列之(1)简单理解梯度下降
        深度学习（三十六）异构计算CUDA学习笔记（1）
        深度学习（三十五）异构计算GLSL学习笔记（1）
        深度学习（三十四）对抗自编码网络-未完待续
        深度学习（三十三）CRF as RNN语义分割-未完待续
        深度学习（三十二）半监督阶梯网络学习笔记-NIPS 2015
        深度学习（三十一）基于深度矩阵分解的属性表征学习
        深度学习（三十）贪婪深度字典学习
        深度学习（二十九）Batch Normalization 学习笔记
        深度学习（二十八）基于多尺度深度网络的单幅图像深度估计-NIPS 2014
        深度学习（二十七）可视化理解卷积神经网络-ECCV 2014
        深度学习（二十六）Network In Network学习笔记-ICLR 2014
        深度学习（二十五）基于Mutil-Scale CNN的图片语义分割、法向量估计-ICCV 2015
        深度学习（二十四）矩阵分解之基于k-means的特征表达学习
        深度学习（二十三）Maxout网络学习-ICML 2013
        深度学习（二十二）Dropout浅层理解与实现
        深度学习（二十一）基于FCN的图像语义分割-CVPR 2015
        深度学习（二十）基于Overfeat的图片分类、定位、检测-2014 ICLR
        深度学习（十九）基于空间金字塔池化的卷积神经网络物体检测-ECCV 2014
        深度学习（十八）基于R-CNN的物体检测-CVPR 2014
        深度学习（十七）基于改进Coarse-to-fine CNN网络的人脸特征点定位-ICCV 2013
        深度学习（十六）基于2-channel network的图片相似度判别-CVPR 2015
        深度学习（十五）基于级联卷积神经网络的人脸特征点定位-CVPR 2013  
        深度学习（十四）基于CNN的性别、年龄识别
        深度学习（十三）caffe之训练数据格式
        深度学习（十二）从自编码到栈式自编码
        深度学习（十一）RNN入门学习
        深度学习（十）keras学习笔记
        深度学习（九）caffe预测、特征可视化Python接口调用
        深度学习（八）RBM受限波尔兹曼机学习-未完待续
        深度学习（七）caffe源码c++学习笔记
        深度学习（六）caffe入门学习
        深度学习（五）caffe环境搭建
        深度学习（四）卷积神经网络入门学习 2.16
        深度学习（三）theano入门学习
        深度学习（二）theano环境搭建
        深度学习（一）深度学习学习资料
机器学习中的数学班
    第1课 机器学习与数学综述
    机器学习的种类与基本思路，假设函数与损失函数，机器学习与统计学、最优化、微分、矩阵运算的关系
    第2课 微积分
    Taylor展式、梯度下降和牛顿法初步、Jensen不等式
    第3课 概率论与数理统计
    常见分布与共轭分布、切比雪夫不等式、大数定理、中心极限定理
    第4课 参数估计
    矩估计、极大似然估计
    第5课 矩阵基础
    线性映射，线性方程，矩阵基本概念，相似变换，特征向量
    第6课 矩阵进阶
    二次型，对称矩阵对角化，奇异值分解
    第7课 凸优化基础
    优化、凸优化基本概念简介，凸集，凸函数
    第8课 凸优化进阶
    凸优化问题标准形式，对偶问题与KKT条件
    牛顿法，内点法
    第9课 从数学到机器学习分类问题
    机器学习与分类问题，空间切分与决策边界，Softmax与linearSVM，损失函数与最小化
    第10课 优化与统计学习的典型应用：SVM进阶
    最大间隔分类，SVM中的目标函数的优化方法，kernel tricks，soft-hard margin，thinking in SVMs
    
机器学习代码
    https://github.com/wepe/MachineLearning
    http://homepages.inf.ed.ac.uk/rbf/IAPR/researchers/MLPAGES/mlcourses.htm
        总的课程链接
    
lda 学习
    七月算法 http://v.youku.com/v_show/id_XMTI0ODI1MTIzNg==.html?from=s1.8-1-1.2
    七月算法 http://v.youku.com/v_show/id_XMTQ1MTU3MTcyNA==.html?from=y1.2-1-87.3.1-1.1-1-1-0-0
    七月算法 http://v.youku.com/v_show/id_XMTI0ODI1MjcxMg==.html?from=s1.8-1-1.2
自然语言处理之LDA II-Variational EM实现
    http://zhikaizhang.cn/2016/07/10/%E8%87%AA%E7%84%B6%E8%AF%AD%E8%A8%80%E5%A4%84%E7%90%86%E4%B9%8BLDA%20II-Variational%20EM%E5%AE%9E%E7%8E%B0/
EM 算法
    http://v.youku.com/v_show/id_XMTQwMjk4MDMzNg==.html?from=y1.2-1-87.3.5-1.1-1-1-4-0
        
异步社区
    http://www.epubit.com.cn/
    
        
基于深度学习的智能问答 作者：周小强 陈清财 曾华军

Cronhub 开源的时间调度系统-马晨

UFLDL Tutorial 无监督特征学习和深度学习的主要观点
    http://deeplearning.stanford.edu/wiki/index.php/UFLDL_Tutorial
    稀疏自编码器
    矢量化编程实现
    预处理：主成分分析与白化
    Softmax回归
    自我学习与无监督特征学习
    建立分类用深度网络
    自编码线性解码器
    处理大型图像
    混杂的
    混杂的主题
    稀疏编码
    独立成分分析样式建模
    其它

    笔记
        http://blog.csdn.net/zhoubl668/article/details/24800611
        DeepLearning学习随记（一）稀疏自编码器
        DeepLearning学习随记（二）Vectorized、PCA和Whitening
        Deep Learning 学习随记（三）Softmax regression - bzjia
        Deep Learning 学习随记（四）自学习和非监督特征学习
        DeepLearning学习随记（五）Deepnetwork深度网络
        Deep Learning 学习随记（六）Linear Decoder 线性解码
        Deep Learning 学习随记（七）Convolution and Pooling --卷积和池化
        Deep Learning 学习随记（八）CNN（Convolutional neural network）理解

    
CSDN
    http://my.csdn.net/wojiushiwo987     铭毅天下
    http://my.csdn.net/zhoubl668        beck_zhou
    

神经网络与深度学习
    作者：邱锡鹏  微博：@邱锡鹏
    绪论 1.13
    数学基础
    机器学习概述
    感知器 1.18
    前馈神经网络
    卷积神经网络 1.17
    循环神经网络
    注意力机制与外部记忆
    生成对抗网络
    词嵌入与语言模型

[Coursera] Neural Networks for Machine Learning — Geoffrey Hinton 2016
    https://www.youtube.com/playlist?list=PLoRl3Ht4JOcdU872GhiYWf6jwrk_SNhz9

KM平台 http://km.oa.com/?kmref=km_header
    腾讯大讲堂举办了“腾讯精准推荐”系列讲座
        再谈推荐-腾讯精准推荐系列”之综述篇
        “腾讯精准推荐系列”之算法篇
        “腾讯精准推荐系列”—系统篇
        “腾讯精准推荐系列”最后一站--数据篇

    QQ游戏智能推荐系统:云开月明GIR(Game Intelligent recommendation)实现详解
        云开月明系统之月明篇（1）—智能推荐—协同过滤
        云开月明系统之月明篇（2）—智能推荐—基于内容的推荐
        云开月明系统之月明篇（3）—智能推荐—社会化推荐
        云开月明系统之云开篇（1）—智能投放—Lookalike audiences
        云开月明系统之云开篇（2）—智能投放—自动化投放功能
        云开月明系统之云开篇（3）—智能投放—产品聚类算法

百度PaddlePaddle开源团队
    http://gitbook.cn/books/588428d95a5adc3f0316026d/index.html
    深度学习第一课
    
    
知识表示学习研究进展


A Survey on Algorithms of the Regularized Convex Optimization Problem

统计学基础
    《概率与数理统计》

编程
    C/C++
    Java
    Python
    Linux Shell

数据挖掘
    数据挖掘：概念与技术
    数据挖掘：实用机器学习工具与技术
    数据挖掘导论
    数据挖掘与R语言
    大数据技术丛书·数据挖掘：实用案例分析（附光盘）
机器学习
    神经网络与机器学习
    机器学习基础教程
    机器学习：实用案例解析
    机器学习实践指南

推荐系统
    
    
模式分类
    模式分类（原书第二版）
大数据
    hadoop
    hive
    hbase
    分布式内存数据
    storm
    spark
自然语言处理
    
算法：
    分类和回归（大多数的分类和回归模型都是有监督的学习或者半监督的学习。）
        Liner  regression
        Logistic regression
        Ridge regression
        Lasso
        Linear Discriminant Analysis
        Basis Expansions，Smoothing Splines
        Kernel Methods
        Additive Models
        Trees
            Classification trees
            Random Forest
        Nearest-Neighbors Methods
        SVM
    求解方法
        Maximum Likelihood Estimation (MLE, 最大似然估计)
        Maximum A Posteriori (MAP, 最大后验估计)
        Least Square (LS, 最小二乘法)
        Maximum Likelihood (ML, 极大似然估计)
        Expectation Maximization (EM)
        Gradient Descent (梯度下降法)
            Classic Newton methods
            Quasi-Newton Methods
            梯度下降法
            批梯度下降法
            增量梯度下降
    无监督的学习
        
    数据加工方法
        收缩法
            Ridge regression
            Lasso
        降维法
            PCA
        ICA特征提取的因子分析
    数据采样方法
        
    神经网络
    
    聚类、回归、决策树、PCA、LSA、LDA、DNN、协同过滤
    
    ACM
        分治与二分查找(8)
        动态规划(36)
        搜索DFS/BFS(22)
        数学题(16)
        Hash(1)
        大数(2)
        字符串处理(13)
        链表(7)
        数论与数组(5)
        树与图论(18)
        栈和队列(7)
        排序(4)
        模拟题(16)
        网络流(0)
        贪心(12)
        线段树(0)
        计算几何(0)
        递归(19)
        位运算与高精度(2)
        排列与组合(4)
        双指针(7)

    C/C++(10)
    JAVA(12)
    Matlab(3)
    PHP/Python/Perl(1)
    
    编程竞赛(6)
    编程面试题(5)

    数据挖掘(19)
    机器学习(11)
    NLP/IR(15)
    OMSA(3)
    PGM/Topic Model(7)
    Math(5)
    PRML(1)
    Hadoop/MapReduce(0)
    Linux/Unix(2)
    游戏研发(1)
    网络安全(4)
    生活相关(5)

数据从业人员图灵参考图书一览表
▌​基础知识

{数学与算法}

    具体数学
    概率论及其应用
    程序员的数学
    程序员的数学2：概率统计
    程序员的数学3：线性代数

    算法（第4版）
    算法基础（第5版）
    计算机程序设计艺术

{编程语言}

SQL

    SQL必知必会（第4版）
    SQL基础教程

    Python与R

    Python编程：从入门到实践
    更多Python图书参考：《图灵Python图书一览表》

学习R
    R语言入门与实践
    R语言实战（第2版）
    R包开发
Java与Scala
    Java技术手册（第6版）
    Java 8实战
    更多Java图书参考：《图灵Java图书》
Scala程序设计（第2版）
    Scala与Clojure函数式编程模式：Java虚拟机高效编程
Go
    Go语言编程
    Go并发编程实战
▌​核心技能
{数据科学}
    数据科学入门
    数据科学实战
    命令行中的数据科学
{分布式框架}
    Spark快速大数据分析
    Spark高级数据分析
    Spark最佳实践
    Hadoop基础教程
    精通Hadoop
{机器学习}
    机器学习
    图解机器学习
    机器学习实战
    机器学习系统设计
    Spark机器学习
    机器学习实践：测试驱动的开发方法
    Mahout实战
    推荐系统
    推荐系统实践
{数据挖掘}
    大数据：互联网大规模数据挖掘与分布式处理（第2版）
    数据挖掘导论（完整版）
    Python数据挖掘入门与实践
{交叉应用}
    精益数据分析
    社会媒体挖掘
    Python网络数据采集
    干净的数据：数据清洗入门与实践
    Pyhton计算机视觉编程
    Elasticsearch服务器开发（第2版）
{数据可视化}
    人人都是数据分析师：Tableau应用实战
    洞悉数据：用可视化方法发掘数据真意
    鲜活的数据：数据可视化指南
    数据可视化实战：使用D3设计交互式图表
▌​云计算

OpenStack
    OpenStack部署实践（第2版）
    OpenStack运维指南
Docker
    Docker开发实践
    Docker容器与容器云
    Docker基础与实战
网络
    腾云：云计算和大数据时代网络技术揭秘
    软件定义网络：SDN与OpenFlow解析
    云数据中心网络技术
    
    
    
    