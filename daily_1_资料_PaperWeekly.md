Autoencoder
    A Hierarchical Neural Autoencoder for Paragraphs and Documents
CNN
    Recurrent Convolutional Neural Networks for Text Classification
    Convolutional Neural Networks for Sentence Classification 
DQN
    Deep Reinforcement Learning for Dialogue Generation #PaperWeekly#
    2016-06-28
    Deep Reinforcement Learning with a Natural Language Action Space #PaperWeekly#
    2016-06-28
    Generating Text with Deep Reinforcement Learning #PaperWeekly#
    2016-06-27
    Language Understanding for Text-based Games using Deep Reinforcement Learning #PaperWeekly#
Memory Network1
    THE GOLDILOCKS PRINCIPLE: READING CHILDREN’S BOOKS WITH EXPLICIT MEMORY REPRESENTATIONS #PaperWeekly#
NLP2
PaperWeekly101
RNN1
RNNLM1
ROUGE1
RSarXiv1
Reading Comprehension6
Representation1
Text Comprehension1
api.ai1
arXiv2
arxiv2
attention3
bot21
chatbot2
dataset1
deep learning1
deeplearning1
language model1
nlp119
open source1
paper7
paperweekly2
reading comprehension1
reinforcement learning1
sentence representations1
seq2seq17
text comprehension1
torch1
word embedding2
word embeddings1
word2vec1
创业1
招聘1
推荐系统2
综述1
自动文摘16
随笔

http://rsarxiv.github.io/


PaperWeekly 第二十一期
PaperWeekly 第二十期
PaperWeekly 第十九期 --- 新文解读（情感分析、机器阅读理解、知识图谱、文本分类）
原创2016-12-24paperweeklyPaperWeekly
引
    本期的PaperWeekly一共分享四篇最近arXiv上发布的高质量paper，包括：情感分析、机器阅读理解、知识图谱、文本分类。人工智能及其相关研究日新月异，本文将带着大家了解一下以上四个研究方向都有哪些最新进展。四篇paper分别是：

    1、Linguistically Regularized LSTMs for Sentiment Classification, 2016.11
    2、End-to-End Answer Chunk Extraction and Ranking for Reading Comprehension, 2016.10
    3、Knowledge will Propel Machine Understanding of Content: Extrapolating from Current Examples, 2016.10
    4、AC-BLSTM: Asymmetric Convolutional Bidirectional LSTM Networks for Text Classification, 2016.11

PaperWeekly 第十八期 --- 提高seq2seq方法所生成对话的流畅度和多样性
原创2016-12-17paperweeklyPaperWeekly
引言
    对话系统是当前的研究热点，也是风险投资的热点，从2016年初开始，成立了无数家做chatbot、语音助手等类似产品的公司，不管是对用户的，还是对企业的，将对话系统这一应用推到了一个新的高度。seq2seq是当前流行的算法框架，给定一个输入，模型自动给出一个不错的输出，听起来都是一件美好的事情。seq2seq在对话系统中的研究比较多，本期PaperWeekly分享4篇的paper notes，涉及到如何提高所生成对话的流畅度和多样性，使得对话系统能够更加接近人类的对话。4篇paper如下：

    1、Sequence to Backward and Forward Sequences: A Content-Introducing Approach to Generative Short-Text Conversation, 2016
    2、A Simple, Fast Diverse Decoding Algorithm for Neural Generation, 2016
    3、DIVERSE BEAM SEARCH: DECODING DIVERSE SOLUTIONS FROM NEURAL SEQUENCE MODELS, 2016
    4、A Diversity-Promoting Objective Function for Neural Conversation Models, 2015
PaperWeekly 第二期 2016-08-16

本周值得读(2017.01.09-2017.01.13)
本周值得读(2016.12.26-2017.01.06)
本周值得读(2016.12.19-2016.12.23)
本周值得读(2016.12.12-2016.12.16)
本周值得读(2016.12.05-2016.12.09)
本周值得读(2016.11.28-2016.12.02)
本周值得读(2016.11.21-2016.11.25)

近期对话系统领域高质量paper汇总
    DIALOG CONTEXT LANGUAGE MODELING WITH RECURRENT NEURAL NETWORKS
        In this work, we propose contextual language models that incorporate dialog level discourse information into language modeling. Previous works on contextual language model treat preceding utterances as a sequence of inputs, without considering dialog interactions. We design recurrent neural network (RNN) based contextual language models that specially track the interactions between speakers in a dialog. Experiment results on Switchboard Dialog Act Corpus show that the proposed model outperforms conventional single turn based RNN language model by 3.3% on perplexity. The proposed models also demonstrate advantageous performance over other competitive contextual language models.
    A Copy-Augmented Sequence-to-Sequence Architecture Gives Good Performance on Task-Oriented Dialogue
        Task-oriented dialogue focuses on conversational agents that participate in user-initiated dialogues on domain-specific topics. In contrast to chatbots, which simply seek to sustain open-ended meaningful discourse, existing task-oriented agents usually explicitly model user intent and belief states. This paper examines bypassing such an explicit representation by depending on a latent neural embedding of state and learning selective attention to dialogue history together with copying to incorporate relevant prior context. We complement recent work by showing the effectiveness of simple sequence-to-sequence neural architectures with a copy mechanism. Our model outperforms more complex memory-augmented models by 7% in per-response generation and is on par with the current state-of-the-art on DSTC2.
    Learning Through Dialogue Interactions
        A good dialogue agent should have the ability to interact with users by both responding to questions and by asking questions, and importantly to learn from both types of interaction. In this work, we explore this direction by designing a simulator and a set of synthetic tasks in the movie domain that allow such interactions between a learner and a teacher. We investigate how a learner can benefit from asking questions in both offline and online reinforcement learning settings, and demonstrate that the learner improves when asking questions. Finally, real experiments with Mechanical Turk validate the approach. Our work represents a first step in developing such end-to-end learned interactive dialogue agents.
    Online Sequence-to-Sequence Active Learning for Open-Domain Dialogue Generation
        We propose an online, end-to-end, neural generative conversational model for open-domain dialog. It is trained using a unique combination of offline two-phase supervised learning and online human-in-the-loop active learning. While most existing research proposes offline supervision or hand-crafted reward functions for online reinforcement, we devise a novel interactive learning mechanism based on a diversity-promoting heuristic for response generation and one-character user-feedback at each step. Experiments show that our model inherently promotes the generation of meaningful, relevant and interesting responses, and can be used to train agents with customized personas, moods and conversational styles.
    Two are Better than One: An Ensemble of Retrieval- and Generation-Based Dialog Systems
        Open-domain human-computer conversation has attracted much attention in the field of NLP. Contrary to rule- or template-based domain-specific dialog systems, open-domain conversation usually requires data-driven approaches, which can be roughly divided into two categories: retrieval-based and generation-based systems. Retrieval systems search a user-issued utterance (called a query) in a large database, and return a reply that best matches the query. Generative approaches, typically based on recurrent neural networks (RNNs), can synthesize new replies, but they suffer from the problem of generating short, meaningless utterances. In this paper, we propose a novel ensemble of retrieval-based and generation-based dialog systems in the open domain. In our approach, the retrieved candidate, in addition to the original query, is fed to an RNN-based reply generator, so that the neural model is aware of more information. The generated reply is then fed back as a new candidate for post-reranking. Experimental results show that such ensemble outperforms each single part of it by a large margin.
            
            
            
            