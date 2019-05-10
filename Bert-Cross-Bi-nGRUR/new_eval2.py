# -*- coding:utf-8 -*-
import tensorflow as tf
import pandas as pd

## define lstm model and reture related features

def convet_to_pos(x,margin=0.15):
    a = (1+margin)/2
    b = (1-margin)/2
    y = a*x + b*tf.abs(x)
    return y

def similarity(out_q,ans_q):
    normalized_q = tf.nn.l2_normalize(out_q, dim=1)
    normalized_a = tf.nn.l2_normalize(ans_q, dim=1)
    out_similarity = tf.reduce_sum(tf.multiply(normalized_q, normalized_a),axis=1)
    # out_similarity = convet_to_pos(tf.reduce_sum(tf.multiply(normalized_q, normalized_a), axis=1))
    return out_similarity

def cos_fun(margin,cand_si,neg_si,batch_size):
    zero = tf.constant(0, shape=[batch_size], dtype=tf.float32)
    margin = tf.constant(margin, shape=[batch_size], dtype=tf.float32)
    losses = tf.maximum(zero, tf.subtract(margin, tf.subtract(cand_si, neg_si)))
    correct = tf.equal(zero, losses)
    acc = tf.reduce_mean(tf.cast(correct, "float"), name="acc")
    # loss = tf.reduce_sum(losses)
    loss = tf.reduce_mean(losses)
    return loss,acc
def max_pooling(input):
    height,width = int(input.get_shape()[1]),int(input.get_shape()[2])
    input = tf.expand_dims(input,-1)
    output = tf.nn.max_pool(
        input,
        ksize=[1,height,1,1],
        strides=[1,1,1,1],
        padding='VALID'
    )
    output = tf.reshape(output,shape=[-1,width])
    return output

def map_mrr_yahoo(similarity,label,question_len):
    MRR = 0
    position = 0
    P_1 = 0
    for k in range(len(question_len)):
        sim_batch = similarity[position:position+question_len[k]]
        lab_batch = label[position:position+question_len[k]]
        position = position+question_len[k]
        matri = [sim_batch,lab_batch]
        matri = zip(*matri)
        matri_sort = sorted(matri,reverse=True)
        for i,j in enumerate(matri_sort):
            if j[1] == 1 :
                MRR_p = 1.0 / (i + 1)
                if i==0:
                    P_1 += 1
                break
        MRR = MRR + MRR_p
    MRR = MRR/len(question_len)
    P_1 = float(P_1)/len(question_len)
    return P_1,MRR

