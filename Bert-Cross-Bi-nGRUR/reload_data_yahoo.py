# -*- coding:utf-8 -*-
import tensorflow as tf
import pickle
from tensorflow.contrib import learn as tf_learn

path = '../yahoo/yahoo_data/data'

def train_data():
    lines = []
    with open("{}/yahoo_train.pkl".format(path),'rb') as f1:
        data = pickle.load(f1)
    question_p = data[0]
    can_ans_p = data[1]
    neg_ans_p = data[2]
    for i in range(len(question_p)):
        lines.append([[question_p[i], "1"],[can_ans_p[i], "1"], [neg_ans_p[i]]])
    return lines

def test_data():
    lines = []
    with open("{}/yahoo_test.pkl".format(path), 'rb') as f1:
        data = pickle.load(f1)
    question_p = data[0]
    can_ans_p = data[1]
    for i in range(len(question_p)):
        lines.append([[question_p[i], "1"], [can_ans_p[i], "1"], [can_ans_p[i], "1"]])
    return lines

def test_data_label():
    with open("{}/yahoo_test.pkl".format(path), 'rb') as f1:
        data = pickle.load(f1)
    label = data[2]
    question_len = data[3]
    return [label,question_len]


def dev_data():
    lines = []
    with open("{}/yahoo_dev.pkl".format(path), 'rb') as f1:
        data = pickle.load(f1)
    question_p = data[0]
    can_ans_p = data[1]
    for i in range(len(question_p)):
        lines.append([[question_p[i], "1"], [can_ans_p[i], "1"], [can_ans_p[i], "1"]])
    return lines

def dev_data_label():
    with open("{}/yahoo_dev.pkl".format(path), 'rb') as f1:
        data = pickle.load(f1)
    label = data[2]
    question_len = data[3]
    return [label,question_len]

