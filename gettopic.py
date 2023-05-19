#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/11/2 9:59
# @Author : Haoxuan Zhang
# Get topic similarity
from bertopic import BERTopic
# see Bertopic's homepage for training topic model
# 加载模型
topic_model = BERTopic.load("model/result_model_new1")

#print(topic_model.get_topic_info())

# 读取数据
f = open(r'alldata/data/result.txt', 'r', encoding='utf-8')
listdata = []
for line in f:
    line = line.strip('\n')
    listdata.append(line)

# 求出主题相似度
for i in listdata:
    similar_topics, similarity = topic_model.find_topics(i, top_n=7)
    doc2 = open('result/result_topic.txt', 'a', encoding='utf-8')
    print(str(similarity), file=doc2)