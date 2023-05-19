#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/7/29 10:19
# @Author : Haoxuan Zhang
# semantic novelty measurement
from sentence_transformers import SentenceTransformer
import scipy.spatial

if __name__ == '__main__':
    model = SentenceTransformer('all-MiniLM-L6-v2', device="cuda:0")

    # corpus data
    l = []
    with open("data/result.txt", 'r', encoding='utf-8') as f:
        l = f.readlines()
        l = [i.rstrip().split('\n') for i in l]

    corpusdata = []

    for i in range(len(l)):
        for j in range(len(l[i])):
            corpusdata.append(l[i][j])
    corpus_embeddings = model.encode(corpusdata)

    # testdata
    f = open('testdata/r.txt', 'r', encoding='utf-8')
    listdata = []
    for line in f:
        line = line.strip('\n')
        listdata.append(line)

    doc = open("result/r.txt", 'a', encoding='utf-8')

    queries = listdata
    query_embeddings = model.encode(queries)


    for query, query_embedding in zip(queries, query_embeddings):
        # closest_n = 10
        # score1 = 0
        score2 = 0
        distance = scipy.spatial.distance.cdist([query_embedding], corpus_embeddings, "cosine")[0]
        result = zip(range(len(distance)), distance)
        result = sorted(result, key=lambda x: x[1])
        # print("--------------------------------------------------------------------------------------")
        # print("Query:", query)
        # print("Result:Top 15 most similar sentences in corpus:")
        # for idx, distance in result[0:closest_n]:
        #   print(corpus[idx].strip(), "(Score:%.8f)" % (1 - distance))
        #   score1 += (1 - distance)
        for idx, distance in result:
            score2 += (1 - distance)

        # print("---------------------------------------------------------------------------------------")
        # print("Score in Close_n: ", score1 / closest_n)

        print(1-(score2 / len(result)), file=doc)
