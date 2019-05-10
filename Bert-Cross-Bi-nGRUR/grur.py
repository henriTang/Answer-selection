# -*- coding:utf-8 -*-
import tensorflow as tf
import new_eval2 as new_eval
from tensorflow.contrib import layers

regularizer = layers.l1_l2_regularizer(scale_l1=1e-6, scale_l2=1e-6)

def S_matri(x1,x2):
    normalized_q = tf.nn.l2_normalize(x1, dim=2)
    normalized_a = tf.nn.l2_normalize(x2, dim=2)
    matri = tf.matmul(normalized_q,tf.transpose(normalized_a, perm=[0,2,1]))
    return matri


class GRU_first(object):
    def __init__(self, input, n_output, n_skip, batch_size):
        self.xt_ini = input
        self.batch_size = batch_size
        self.time_step = int(self.xt_ini.get_shape()[1])
        self.n_input = int(self.xt_ini.get_shape()[2])
        self.n_output = n_output
        with tf.variable_scope("gru_q_a"):
            self.skip_Wr = tf.get_variable(shape=[self.n_input, self.n_output], name="skip_Wr",regularizer=regularizer)
            self.skip_Ur = tf.get_variable(shape=[self.n_output, self.n_output], name="skip_Ur",regularizer=regularizer)
            self.skip_br = tf.get_variable(name='skip_br', initializer=tf.zeros([self.n_output]),regularizer=regularizer)
            self.skip_W = tf.get_variable(shape=[self.n_input, self.n_output], name="skip_W",regularizer=regularizer)
            self.skip_U = tf.get_variable(shape=[self.n_output, self.n_output], name="skip_U",regularizer=regularizer)
            self.skip_b = tf.get_variable(name='skip_b', initializer=tf.zeros([self.n_output]),regularizer=regularizer)
            self.skip_Wz = tf.get_variable(shape=[self.n_input, self.n_output], name="skip_Wz",regularizer=regularizer)
            self.skip_Uz = tf.get_variable(shape=[self.n_output, self.n_output], name="skip_Uz",regularizer=regularizer)
            self.skip_bz = tf.get_variable(name='skip_bz', initializer=tf.zeros([self.n_output]),regularizer=regularizer)

            self.ht1 = tf.zeros(shape=[self.batch_size, self.n_output],dtype=tf.float32)
            self.Wr = tf.get_variable(shape=[self.n_input, self.n_output], name="Wr",regularizer=regularizer)
            self.Ur = tf.get_variable(shape=[self.n_output, self.n_output], name="Ur",regularizer=regularizer)
            self.br = tf.get_variable(name='br',initializer=tf.zeros([self.n_output]),regularizer=regularizer)
            self.W = tf.get_variable(shape=[self.n_input, self.n_output], name="W",regularizer=regularizer)
            self.U = tf.get_variable(shape=[self.n_output, self.n_output], name="U",regularizer=regularizer)
            self.b = tf.get_variable(name='b', initializer=tf.zeros([self.n_output]),regularizer=regularizer)
            self.Wz = tf.get_variable(shape=[self.n_input,self.n_output], name="Wz",regularizer=regularizer)
            self.Uz = tf.get_variable(shape=[self.n_output, self.n_output], name="Uz",regularizer=regularizer)
            self.bz = tf.get_variable(name='bz', initializer=tf.zeros([self.n_output]))
            self.xt = tf.unstack(self.xt_ini,None,1)

        skip_cell = [tf.zeros(shape=[self.batch_size, self.n_output], dtype=tf.float32)]*n_skip

        for step in range(self.time_step):
            time_x = self.xt[step]
            skip_r = tf.nn.sigmoid(tf.matmul(time_x, self.skip_Wr) + tf.matmul(skip_cell[-1], self.skip_Ur)
                                   + self.skip_br)
            skip_h = tf.nn.tanh( tf.matmul(time_x, self.skip_W) +
                                 tf.matmul(tf.multiply(skip_r, skip_cell[-1]), self.skip_U) + self.skip_b)
            skip_z = tf.nn.sigmoid(tf.matmul(time_x, self.skip_Wz)
                                   + tf.matmul(skip_cell[-1], self.skip_Uz)+self.skip_bz)

            rt = tf.nn.sigmoid(tf.matmul(time_x, self.Wr) + tf.matmul(self.ht1, self.Ur) + self.br)
            h_ = tf.nn.tanh(
                tf.matmul(time_x, self.W) + tf.matmul(tf.multiply(rt, self.ht1), self.U) + self.b)
            zt = tf.nn.sigmoid(tf.matmul(time_x, self.Wz) + tf.matmul(self.ht1, self.Uz)+self.bz)

            h_ = tf.multiply(skip_h, tf.subtract(tf.ones(shape=skip_z.get_shape()), skip_z)) + \
                 tf.multiply(h_, skip_z)

            ht = tf.multiply(self.ht1,tf.subtract(tf.ones(shape=zt.get_shape()),zt)) + tf.multiply(h_, zt)
            self.ht1 = ht

            if step == 0:
                self.output_p = []
                self.output_p.append(ht)
            else:
                self.output_p.append(ht)
            skip_cell.pop()
            skip_cell.insert(0,ht)

        self.output = tf.stack(self.output_p,axis=1)
        self.ht1 = tf.zeros(shape=[self.batch_size, self.n_output], dtype=tf.float32)

class BiGRU_first(object):
    def __init__(self,input,n_output,n_skip,batch_size):
        with tf.variable_scope("FW"):
            fw = GRU_first(input, n_output,n_skip, batch_size)
            self.fw_output = fw.output
        with tf.variable_scope("BW"):
            input = tf.reverse(input, [True])
            bw = GRU_first(input, n_output,n_skip, batch_size)
            self.bw_output = tf.reverse(bw.output, [True])
        self.output_ori = tf.concat([self.fw_output, self.bw_output], 2)
        self.output = new_eval.max_pooling(self.output_ori)

class GRU_q_a(object):
    def __init__(self, input, n_output,cnn_out, n_skip, batch_size):
        self.xt_ini = input
        self.batch_size = batch_size
        self.time_step = int(self.xt_ini.get_shape()[1])
        self.n_input = int(self.xt_ini.get_shape()[2])
        self.n_output = n_output

        with tf.variable_scope("gru_q_a"):
            self.skip_Wr = tf.get_variable(shape=[self.n_input, self.n_output], name="skip_Wr",regularizer=regularizer)
            self.skip_Ur = tf.get_variable(shape=[self.n_output, self.n_output], name="skip_Ur",regularizer=regularizer)
            self.skip_br = tf.get_variable(name='skip_br', initializer=tf.zeros([self.n_output]),regularizer=regularizer)
            self.skip_W = tf.get_variable(shape=[self.n_input, self.n_output], name="skip_W",regularizer=regularizer)
            self.skip_U = tf.get_variable(shape=[self.n_output, self.n_output], name="skip_U",regularizer=regularizer)
            self.skip_b = tf.get_variable(name='skip_b', initializer=tf.zeros([self.n_output]),regularizer=regularizer)
            self.skip_Wz = tf.get_variable(shape=[self.n_input, self.n_output], name="skip_Wz",regularizer=regularizer)
            self.skip_Uz = tf.get_variable(shape=[self.n_output, self.n_output], name="skip_Uz",regularizer=regularizer)
            self.skip_bz = tf.get_variable(name='skip_bz', initializer=tf.zeros([self.n_output]),regularizer=regularizer)

            self.W_cnn_r = tf.get_variable(shape=[int(cnn_out.get_shape()[1]), self.n_output], name="W_cnn_q_ar",regularizer=regularizer)
            self.W_cnn = tf.get_variable(shape=[int(cnn_out.get_shape()[1]), self.n_output], name="W_cnn_q_a",regularizer=regularizer)
            self.W_cnn_z = tf.get_variable(shape=[int(cnn_out.get_shape()[1]), self.n_output], name="W_cnn_q_az",regularizer=regularizer)
            self.cnn = tf.tile(tf.expand_dims(cnn_out, 1), [1, self.time_step, 1])
            self.cnn_t = tf.unstack(self.cnn, None, 1)

            self.ht1 = tf.zeros(shape=[self.batch_size, self.n_output],dtype=tf.float32)
            self.Wr = tf.get_variable(shape=[self.n_input, self.n_output], name="Wr",regularizer=regularizer)
            self.Ur = tf.get_variable(shape=[self.n_output, self.n_output], name="Ur",regularizer=regularizer)
            self.br = tf.get_variable(name='br',initializer=tf.zeros([self.n_output]),regularizer=regularizer)
            self.W = tf.get_variable(shape=[self.n_input, self.n_output], name="W",regularizer=regularizer)
            self.U = tf.get_variable(shape=[self.n_output, self.n_output], name="U",regularizer=regularizer)
            self.b = tf.get_variable(name='b', initializer=tf.zeros([self.n_output]),regularizer=regularizer)
            self.Wz = tf.get_variable(shape=[self.n_input,self.n_output], name="Wz",regularizer=regularizer)
            self.Uz = tf.get_variable(shape=[self.n_output, self.n_output], name="Uz",regularizer=regularizer)
            self.bz = tf.get_variable(name='bz', initializer=tf.zeros([self.n_output]))
            self.xt = tf.unstack(self.xt_ini,None,1)


        skip_cell = [tf.zeros(shape=[self.batch_size, self.n_output],dtype=tf.float32)]*n_skip

        for step in range(self.time_step):
            time_x = self.xt[step]
            time_cnn = self.cnn_t[step]

            skip_r = tf.nn.sigmoid(tf.matmul(time_x, self.skip_Wr) + tf.matmul(skip_cell[-1], self.skip_Ur)
                                   + self.skip_br)
            skip_h = tf.nn.tanh( tf.matmul(time_x, self.skip_W) +
                                 tf.matmul(tf.multiply(skip_r, skip_cell[-1]), self.skip_U) + self.skip_b)
            skip_z = tf.nn.sigmoid(tf.matmul(time_x, self.skip_Wz)
                                   + tf.matmul(skip_cell[-1], self.skip_Uz)+self.skip_bz)

            rt = tf.nn.sigmoid(tf.matmul(time_x, self.Wr) + tf.matmul(self.ht1, self.Ur) + tf.matmul(time_cnn,self.W_cnn_r)+self.br)
            h_ = tf.nn.tanh(
                tf.matmul(time_x, self.W) + tf.matmul(tf.multiply(rt, self.ht1), self.U) +tf.matmul(time_cnn,self.W_cnn)+ self.b)
            zt = tf.nn.sigmoid(tf.matmul(time_x, self.Wz) + tf.matmul(self.ht1, self.Uz)+tf.matmul(time_cnn,self.W_cnn_z)+self.bz)

            h_ = tf.multiply(skip_h, tf.subtract(tf.ones(shape=skip_z.get_shape()), skip_z)) + \
                 tf.multiply(h_, skip_z)

            ht = tf.multiply(self.ht1,tf.subtract(tf.ones(shape=zt.get_shape()),zt)) + tf.multiply(h_, zt)
            self.ht1 = ht

            if step == 0:
                self.output_p = []
                self.output_p.append(ht)
            else:
                self.output_p.append(ht)
            skip_cell.pop()
            skip_cell.insert(0,ht)

        self.output = tf.stack(self.output_p,axis=1)
        self.ht1 = tf.zeros(shape=[self.batch_size, self.n_output], dtype=tf.float32)

class BiGRU_q_a(object):
    def __init__(self,input, n_output,cnn_out, n_skip,batch_size):
        with tf.variable_scope("FW"):
            fw = GRU_q_a(input, n_output,cnn_out,n_skip,batch_size)
            self.fw_output = fw.output
        with tf.variable_scope("BW"):
            input = tf.reverse(input, [True])
            bw = GRU_q_a(input, n_output,cnn_out,n_skip,batch_size)
            self.bw_output = tf.reverse(bw.output, [True])
        self.output_ori = tf.concat([self.fw_output, self.bw_output], 2)
        self.output = new_eval.max_pooling(self.output_ori)


class CNN_qa(object):
    def __init__(self, input, filter_sizes, CNN_name, num_filters):
        self.input = input
        input_height = int(input.get_shape()[1])
        input_width = int(input.get_shape()[2])
        self.input = tf.expand_dims(self.input, -1)
        pooled_outputs = []
        num_filters_total = num_filters * len(filter_sizes)
        number = 1
        if CNN_name == "q":
            filter_width = input_width
            for filter_height in filter_sizes:
                with tf.variable_scope("conv2-q-{}".format(number)):
                    number += 1
                    filter_shape = [filter_height, filter_width, 1, num_filters]
                    W = tf.get_variable(initializer=tf.truncated_normal(filter_shape, stddev=0.1), name='W',regularizer=regularizer)
                    b = tf.get_variable(initializer=tf.constant(0.1, shape=[num_filters]), name='b',regularizer=regularizer)
                    conv = tf.nn.conv2d(self.input, W, strides=[1, 1, 1, 1], padding='VALID', name='conv')
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
                    pooled = tf.nn.max_pool(h, ksize=[1, input_height - filter_height + 1, input_width-filter_width+1, 1], strides=[1, 1, 1, 1],
                                            padding='VALID', name='pool')
                    pooled_outputs.append(pooled)
        else:
            filter_height = input_height
            for filter_width in filter_sizes:
                with tf.variable_scope("conv2-a-{}".format(number)):
                    number += 1
                    filter_shape = [filter_height, filter_width, 1, num_filters]
                    W = tf.get_variable(initializer=tf.truncated_normal(filter_shape, stddev=0.1), name='W',regularizer=regularizer)
                    b = tf.get_variable(initializer=tf.constant(0.1, shape=[num_filters]), name='b',regularizer=regularizer)
                    conv = tf.nn.conv2d(self.input, W, strides=[1, 1, 1, 1], padding='VALID', name='conv')
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
                    pooled = tf.nn.max_pool(h,
                                            ksize=[1, input_height - filter_height + 1, input_width - filter_width + 1,
                                                   1], strides=[1, 1, 1, 1],
                                            padding='VALID', name='pool')
                    pooled_outputs.append(pooled)
        self.output = tf.reshape(tf.concat(pooled_outputs, 3), [-1, num_filters_total])


class Score_QA(object):
    def __init__(self,q_embedding,a_embedding,q_output,a_output,filter_sizes,num_filters,attention_size,n_output,n_skip,batch_size):
        matrix = S_matri(q_output, a_output)
        with tf.variable_scope("cnn_q"):
            cnn_q = CNN_qa(matrix, filter_sizes, "q", num_filters)
            cnn_q_out = cnn_q.output
        with tf.variable_scope("cnn_a"):
            cnn_a = CNN_qa(matrix, filter_sizes, "a", num_filters)
            cnn_a_out = cnn_a.output

        with tf.variable_scope("second_gru_q_a"):
            gru_q = BiGRU_q_a(q_embedding,n_output,cnn_q_out,n_skip,batch_size)
            self.gru_q_output = gru_q.output
        with tf.variable_scope("second_gru_q_a",reuse=True):
            gru_a = BiGRU_q_a(a_embedding,n_output,cnn_a_out,n_skip,batch_size)
            self.gru_a_output = gru_a.output


class GRU_QA():
    def __init__(self,output_layer1,output_layer2,len_q,n_output,margin,filter_sizes,num_filters,attention_size,n_skip, batch_size):
        self.input_can_q = output_layer1[:,0:len_q+1,:]
        self.input_cand_a = output_layer1[:,len_q+2:,:]
        self.input_neg_q = output_layer2[:,0:len_q+1,:]
        self.input_neg_a = output_layer2[:,len_q+2:,:]
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters

        self.classification_W = tf.get_variable(shape=[n_output*2,1], name="classification_W",regularizer=regularizer)
        self.classification_b = tf.get_variable(shape=[1], name="classification_b", regularizer=regularizer)

        with tf.variable_scope("first_gru_q_a"):
            first_gru_can_q = BiGRU_first(self.input_can_q, n_output, n_skip,batch_size)
            first_gru_can_q_output = first_gru_can_q.output_ori
        with tf.variable_scope("first_gru_q_a", reuse=True):
            first_gru_neg_q = BiGRU_first(self.input_neg_q, n_output, n_skip,batch_size)
            first_gru_neg_q_output = first_gru_neg_q.output_ori
            first_gru_can_a = BiGRU_first(self.input_cand_a, n_output, n_skip,batch_size)
            first_gru_can_a_output = first_gru_can_a.output_ori
            first_gru_neg_a = BiGRU_first(self.input_neg_a, n_output, n_skip,batch_size)
            first_gru_neg_a_output = first_gru_neg_a.output_ori

        with tf.variable_scope("gru_q_a"):
            score_q_a_cand = Score_QA(self.input_can_q,self.input_cand_a,first_gru_can_q_output,first_gru_can_a_output,filter_sizes,num_filters,attention_size,n_output,n_skip,batch_size)
            self.score_cand = new_eval.similarity(score_q_a_cand.gru_q_output,score_q_a_cand.gru_a_output)
            #self.score_cand = tf.nn.sigmoid(tf.matmul(score_q_a_cand.gru_a_output,self.classification_W)+self.classification_b)

        with tf.variable_scope("gru_q_a", reuse=True):
            score_q_a_neg = Score_QA(self.input_neg_q,self.input_neg_a,first_gru_neg_q_output, first_gru_neg_a_output, filter_sizes, num_filters, attention_size,n_output,n_skip,batch_size)
            self.score_neg = new_eval.similarity(score_q_a_neg.gru_q_output, score_q_a_neg.gru_a_output)
            #self.score_neg = tf.nn.sigmoid(tf.matmul(score_q_a_neg.gru_a_output, self.classification_W) + self.classification_b)

        loss, self.acc = new_eval.cos_fun(margin, tf.reshape(self.score_cand,shape=[-1,1]), tf.reshape(self.score_neg,[-1,1]), batch_size)
        self.loss = tf.add(loss, tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)))
