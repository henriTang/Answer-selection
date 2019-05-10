# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import os
import modeling
import optimization
import tokenization
import tensorflow as tf
import time
from datetime import datetime
import pandas as pd
import new_eval2 as new_eval
import reload_data_yahoo
import grur

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "data_dir", "../yahoo/yahoo_data/data",
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "bert_config_file", "./uncased_L-12_H-768_A-12/bert_config.json",
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("task_name", "MRPC", "The name of the task to train.")

flags.DEFINE_string("vocab_file", "./uncased_L-12_H-768_A-12/vocab.txt",
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", "./output_model_yahoo2",
    "The output directory where the model checkpoints will be written.")

## Other parameters

flags.DEFINE_string(
    "init_checkpoint", "./uncased_L-12_H-768_A-12/bert_model.ckpt",
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 103,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool("do_train", True, "Whether to run training.")

flags.DEFINE_bool("do_eval", True, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_predict", False,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 16, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

# flags.DEFINE_integer("save_checkpoints_steps", 1000,
#                      "How often to save the model checkpoint.")
#
# flags.DEFINE_integer("iterations_per_loop", 1000,
#                      "How many steps to make in each estimator call.")

flags.DEFINE_integer("save_checkpoints_steps", 10,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 10,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_string("checkpoint_model_path", "./save_model/yahoo2", "save the model")

flags.DEFINE_float("max_grad_norm", 5, "gradient")
flags.DEFINE_integer("early_stop", 1000, "stop run")
flags.DEFINE_integer("len_q", 50, "question")
flags.DEFINE_integer("len_a", 50, "answer")
flags.DEFINE_integer("batch_size", 23, "batch_size")


# class InputExample(object):
#   """A single training/test example for simple sequence classification."""
#
#   def __init__(self, guid, text_a, text_b=None, text_c=None, label1=None, label2=None):
#     """Constructs a InputExample.
#
#     Args:
#       guid: Unique id for the example.
#       text_a: string. The untokenized text of the first sequence. For single
#         sequence tasks, only this sequence must be specified.
#       text_b: (Optional) string. The untokenized text of the second sequence.
#         Only must be specified for sequence pair tasks.
#       label: (Optional) string. The label of the example. This should be
#         specified for train and dev examples, but not for test examples.
#     """
#     self.guid = guid
#     self.text_a = text_a
#     self.text_b = text_b
#     self.text_c = text_c
#     self.label1 = label1
#     self.label2 = label2

class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, text_a, text_b=None, label=None):
    """Constructs a InputExample.

    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    self.guid = guid
    self.text_a = text_a
    self.text_b = text_b
    self.label = label


class PaddingInputExample(object):
  """Fake example so the num input examples is a multiple of the batch size.

  When running eval/predict on the TPU, we need to pad the number of examples
  to be a multiple of the batch size, because the TPU requires a fixed batch
  size. The alternative is to drop the last batch, which is bad because it means
  the entire output data won't be generated.

  We use this class instead of `None` because treating `None` as padding
  battches could cause silent errors.
  """


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               input_ids,
               input_mask,
               segment_ids,
               label_id,
               is_real_example=True):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.label_id = label_id
    self.is_real_example = is_real_example


class DataProcessor(object):
  """Base class for data converters for sequence classification data sets."""

  def get_train_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the train set."""
    raise NotImplementedError()

  def get_dev_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the dev set."""
    raise NotImplementedError()

  def get_test_examples(self, data_dir):
    """Gets a collection of `InputExample`s for prediction."""
    raise NotImplementedError()

  def get_labels(self):
    """Gets the list of labels for this data set."""
    raise NotImplementedError()

  @classmethod
  def _read_tsv(cls, input_file, quotechar=None):
    """Reads a tab separated value file."""
    pd_file = pd.read_excel(input_file)
    lines = []
    for line in pd_file.values:
       lines.append([str(line[7]), line[1], line[5], line[2], line[6]])
    return lines

class MrpcProcessor(DataProcessor):
  """Processor for the MRPC data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(reload_data_yahoo.train_data(), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(reload_data_yahoo.dev_data(), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(reload_data_yahoo.test_data(), "test")

  def get_labels(self):
    """See base class."""
    return ["0", "1"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      guid = "%s-%s" % (set_type, i)
      text_a = tokenization.convert_to_unicode(line[0][0])
      text_b = tokenization.convert_to_unicode(line[1][0])
      text_c = tokenization.convert_to_unicode(line[2][0])
      label1 = tokenization.convert_to_unicode("1")
      label2 = tokenization.convert_to_unicode("0")
      examples.append(
          [InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label1), InputExample(guid=guid, text_a=text_a, text_b=text_c, label=label2)])
    return examples

def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer):
  """Converts a single `InputExample` into a single `InputFeatures`."""

  if isinstance(example, PaddingInputExample):
    return InputFeatures(
        input_ids=[0] * max_seq_length,
        input_mask=[0] * max_seq_length,
        segment_ids=[0] * max_seq_length,
        label_id=0,
        is_real_example=False)

  label_map = {}
  for (i, label) in enumerate(label_list):
    label_map[label] = i

  tokens_a = tokenizer.tokenize(example.text_a)
  tokens_b = None
  if example.text_b:
    tokens_b = tokenizer.tokenize(example.text_b)

  if tokens_b:
    # Modifies `tokens_a` and `tokens_b` in place so that the total
    # length is less than the specified length.
    # Account for [CLS], [SEP], [SEP] with "- 3"
    _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
  else:
    # Account for [CLS] and [SEP] with "- 2"
    if len(tokens_a) > max_seq_length - 2:
      tokens_a = tokens_a[0:(max_seq_length - 2)]

  # The convention in BERT is:
  # (a) For sequence pairs:
  #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
  #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
  # (b) For single sequences:
  #  tokens:   [CLS] the dog is hairy . [SEP]
  #  type_ids: 0     0   0   0  0     0 0
  #
  # Where "type_ids" are used to indicate whether this is the first
  # sequence or the second sequence. The embedding vectors for `type=0` and
  # `type=1` were learned during pre-training and are added to the wordpiece
  # embedding vector (and position vector). This is not *strictly* necessary
  # since the [SEP] token unambiguously separates the sequences, but it makes
  # it easier for the model to learn the concept of sequences.
  #
  # For classification tasks, the first vector (corresponding to [CLS]) is
  # used as the "sentence vector". Note that this only makes sense because
  # the entire model is fine-tuned.
  tokens = []
  segment_ids = []
  tokens.append("[CLS]")
  segment_ids.append(0)
  for token in tokens_a:
    tokens.append(token)
    segment_ids.append(0)
  tokens.append("[SEP]")
  segment_ids.append(0)

  if tokens_b:
    for token in tokens_b:
      tokens.append(token)
      segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)

  input_ids = tokenizer.convert_tokens_to_ids(tokens)

  # The mask has 1 for real tokens and 0 for padding tokens. Only real
  # tokens are attended to.
  input_mask = [1] * len(input_ids)

  # Zero-pad up to the sequence length.
  while len(input_ids) < max_seq_length:
    input_ids.append(0)
    input_mask.append(0)
    segment_ids.append(0)

  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length

  label_id = label_map[example.label]
  if ex_index < 5:
    tf.logging.info("*** Example ***")
    tf.logging.info("guid: %s" % (example.guid))
    tf.logging.info("tokens: %s" % " ".join(
        [tokenization.printable_text(x) for x in tokens]))
    tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
    tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
    tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
    tf.logging.info("label: %s (id = %d)" % (example.label, label_id))

  feature = InputFeatures(
      input_ids=input_ids,
      input_mask=input_mask,
      segment_ids=segment_ids,
      label_id=label_id,
      is_real_example=True)
  return feature


def file_based_convert_examples_to_features(
    examples, label_list, max_seq_length, tokenizer, output_file):
  """Convert a set of `InputExample`s to a TFRecord file."""

  writer = tf.python_io.TFRecordWriter(output_file)

  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

    feature1 = convert_single_example(ex_index, example[0], label_list,
                                     max_seq_length, tokenizer)
    feature2 = convert_single_example(ex_index, example[1], label_list,
                                     max_seq_length, tokenizer)

    def create_int_feature(values):
      f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
      return f

    features = collections.OrderedDict()
    features["input_ids1"] = create_int_feature(feature1.input_ids)
    features["input_mask1"] = create_int_feature(feature1.input_mask)
    features["segment_ids1"] = create_int_feature(feature1.segment_ids)
    features["label_ids1"] = create_int_feature([feature1.label_id])
    features["is_real_example1"] = create_int_feature([int(feature1.is_real_example)])

    features["input_ids2"] = create_int_feature(feature2.input_ids)
    features["input_mask2"] = create_int_feature(feature2.input_mask)
    features["segment_ids2"] = create_int_feature(feature2.segment_ids)
    features["label_ids2"] = create_int_feature([feature2.label_id])
    features["is_real_example2"] = create_int_feature([int(feature2.is_real_example)])

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(tf_example.SerializeToString())
  writer.close()


class file_based_input_fn_builder:
  def __init__(self, seq_length, drop_remainder, params):
    self.name_to_features = {
      "input_ids1": tf.FixedLenFeature([seq_length], tf.int64),
      "input_mask1": tf.FixedLenFeature([seq_length], tf.int64),
      "segment_ids1": tf.FixedLenFeature([seq_length], tf.int64),
      "label_ids1": tf.FixedLenFeature([], tf.int64),
      "is_real_example1": tf.FixedLenFeature([], tf.int64),
      "input_ids2": tf.FixedLenFeature([seq_length], tf.int64),
      "input_mask2": tf.FixedLenFeature([seq_length], tf.int64),
      "segment_ids2": tf.FixedLenFeature([seq_length], tf.int64),
      "label_ids2": tf.FixedLenFeature([], tf.int64),
      "is_real_example2": tf.FixedLenFeature([], tf.int64)
    }
    self.params = params
    self.drop_remainder = drop_remainder

  def _decode_record(self, record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.to_int32(t)
      example[name] = t

    return example

  def creat_train_dataset(self):
      d = tf.data.TFRecordDataset(self.params["train_file"])
      d = d.repeat(self.params["num_epochs"])
      d = d.shuffle(buffer_size=self.params["shuffle"])
      d = d.apply(
          tf.contrib.data.map_and_batch(
              lambda record: self._decode_record(record, self.name_to_features),
              batch_size=self.params["train_batch_size"],
              drop_remainder=self.drop_remainder))
      d.prefetch(buffer_size=self.params["train_batch_size"])
      return d
  def creat_eval_dataset(self):
      d = tf.data.TFRecordDataset(self.params["eval_file"])
      d = d.repeat(2)
      d = d.apply(tf.contrib.data.map_and_batch(
              lambda record: self._decode_record(record, self.name_to_features),
              batch_size=self.params["eval_batch_size"],
              drop_remainder=self.drop_remainder))
      d.prefetch(buffer_size=self.params["eval_batch_size"])
      return d
  def creat_test_dataset(self):
      d = tf.data.TFRecordDataset(self.params["test_file"])
      d = d.repeat(2)
      d = d.apply(tf.contrib.data.map_and_batch(
              lambda record: self._decode_record(record, self.name_to_features),
              batch_size=self.params["test_batch_size"],
              drop_remainder=self.drop_remainder))
      d.prefetch(buffer_size=self.params["test_batch_size"])
      return d



def _truncate_seq_pair(tokens_a, tokens_b, max_length, len_q=FLAGS.len_q, len_a=FLAGS.len_a):
  """Truncates a sequence pair in place to the maximum length."""

  # This is a simple heuristic which will always truncate the longer sequence
  # one token at a time. This makes more sense than truncating an equal percent
  # of tokens from each, since if one sequence is very short then each token
  # that's truncated likely contains more information than a longer sequence.
  while True:
    if len(tokens_a) == len_q and len(tokens_b) == len_a:
      break
    if len(tokens_a) > len_q:
      tokens_a.pop()
    if len(tokens_a) < len_q:
      tokens_a.append(".")
    if len(tokens_b) > len_a:
      tokens_b.pop()
    if len(tokens_b) < len_a:
        tokens_b.append(".")


def create_model(bert_config, is_training, input_ids1, input_mask1, segment_ids1, label_ids1, input_ids2, input_mask2, segment_ids2, label_ids2,
                 num_labels, use_one_hot_embeddings):
  """Creates a classification model."""
  with tf.variable_scope(""):
      model1 = modeling.BertModel(
          config=bert_config,
          is_training=is_training,
          input_ids=input_ids1,
          input_mask=input_mask1,
          token_type_ids=segment_ids1,
          use_one_hot_embeddings=use_one_hot_embeddings)
  with tf.variable_scope("", reuse=True):
      model2 = modeling.BertModel(
          config=bert_config,
          is_training=is_training,
          input_ids=input_ids2,
          input_mask=input_mask2,
          token_type_ids=segment_ids2,
          use_one_hot_embeddings=use_one_hot_embeddings)

  # In the demo, we are doing a simple classification task on the entire
  # segment.
  #
  # If you want to use the token-level output, use model.get_sequence_output()
  # instead.

  output_layer1 = model1.get_sequence_output()
  output_layer2 = model2.get_sequence_output()
  return (output_layer1, output_layer2)

class only_input_fro_QA:
    def __init__(self):
      self.output_layer1 = tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.max_seq_length, 768])
      self.output_layer2 = tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.max_seq_length, 768])

      GRU_QA = grur.GRU_QA(self.output_layer1, self.output_layer2, FLAGS.len_q, n_output=200, margin=0.15, filter_sizes=[1,2,3,5,7,9], num_filters=100,
                       attention_size=200, n_skip=6, batch_size=FLAGS.batch_size)

      self.loss = GRU_QA.loss
      self.acc = GRU_QA.acc
      self.score_cand = GRU_QA.score_cand
      self.score_neg = GRU_QA.score_neg


class model_fn_builder(object):
  def __init__(self, seq_length, bert_config, num_labels, use_one_hot_embeddings):
    self.input_ids1 = tf.placeholder(dtype=tf.int32, shape=[None, seq_length])
    self.input_mask1 = tf.placeholder(dtype=tf.int32, shape=[None, seq_length])
    self.segment_ids1 = tf.placeholder(dtype=tf.int32, shape=[None, seq_length])
    self.label_ids1 = tf.placeholder(dtype=tf.int32, shape=[None])
    self.is_real_example1 = tf.placeholder(dtype=tf.int32, shape=[None])
    self.input_ids2 = tf.placeholder(dtype=tf.int32, shape=[None, seq_length])
    self.input_mask2 = tf.placeholder(dtype=tf.int32, shape=[None, seq_length])
    self.segment_ids2 = tf.placeholder(dtype=tf.int32, shape=[None, seq_length])
    self.label_ids2 = tf.placeholder(dtype=tf.int32, shape=[None])
    self.is_real_example2 = tf.placeholder(dtype=tf.int32, shape=[None])
    features = {
      "input_ids1": self.input_ids1,
      "input_mask1": self.input_mask1,
      "segment_ids1": self.segment_ids1,
      "label_ids1": self.label_ids1,
      "is_real_example1": self.is_real_example1,
      "input_ids2": self.input_ids2,
      "input_mask2": self.input_mask2,
      "segment_ids2": self.segment_ids2,
      "label_ids2": self.label_ids2,
      "is_real_example2": self.is_real_example2
    }
    self.is_training = tf.placeholder(tf.bool, shape=[])

    # print("*** Features ***")
    # for name in sorted(features.keys()):
    #   print("  name = %s, shape = %s" % (name, features[name].shape))

    input_ids1 = features["input_ids1"]
    input_mask1 = features["input_mask1"]
    segment_ids1 = features["segment_ids1"]
    label_ids1 = features["label_ids1"]

    input_ids2 = features["input_ids2"]
    input_mask2 = features["input_mask2"]
    segment_ids2 = features["segment_ids2"]
    label_ids2 = features["label_ids2"]

    (self.output_layer1, self.output_layer2) = create_model(bert_config, False,input_ids1, input_mask1,
            segment_ids1, label_ids1, input_ids2, input_mask2, segment_ids2, label_ids2, num_labels, use_one_hot_embeddings)


def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer):
  """Convert a set of `InputExample`s to a list of `InputFeatures`."""

  features = []
  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

    feature = convert_single_example(ex_index, example, label_list,
                                     max_seq_length, tokenizer)

    features.append(feature)
  return features


def bert_lr_model(num_epoches, shuffle, train_batch_size, eval_batch_size, test_batch_size, learning_rate, save_frequence, start_eval, early_stop, num_labels=2):
    start1 = time.clock()
    start2 = datetime.now()

    tf.logging.set_verbosity(tf.logging.INFO)

    processors = {
        "mrpc": MrpcProcessor
    }

    tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                  FLAGS.init_checkpoint)
    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
        raise ValueError("At least one of `do_train`, `do_eval` or `do_predict' must be True.")

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError("Cannot use sequence length %d because the BERT model was only trained up to sequence length %d" %
                         (FLAGS.max_seq_length, bert_config.max_position_embeddings))

    tf.gfile.MakeDirs(FLAGS.output_dir)

    task_name = FLAGS.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()

    label_list = processor.get_labels()

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    train_examples = processor.get_train_examples(FLAGS.data_dir)
    train_len = len(train_examples)
    train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
    file_based_convert_examples_to_features(train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file)

    eval_examples = processor.get_dev_examples(FLAGS.data_dir)
    eval_len = len(eval_examples)
    eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
    file_based_convert_examples_to_features(eval_examples, label_list, FLAGS.max_seq_length, tokenizer, eval_file)

    test_examples = processor.get_test_examples(FLAGS.data_dir)
    test_len = len(test_examples)
    test_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
    file_based_convert_examples_to_features(test_examples, label_list, FLAGS.max_seq_length, tokenizer, test_file)

    params = {"train_file": train_file, "eval_file": eval_file, "test_file": test_file, "num_epochs": num_epoches, "shuffle": shuffle,
              "train_batch_size": train_batch_size, "eval_batch_size": eval_batch_size, "test_batch_size": test_batch_size}

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    # sess = tf.Session()
    with sess.as_default():
        data = file_based_input_fn_builder(FLAGS.max_seq_length, False, params)
        data_iter = data.creat_train_dataset()
        iterator = data_iter.make_one_shot_iterator()
        next_element = iterator.get_next()
        sess_bert_lr = model_fn_builder(FLAGS.max_seq_length, bert_config, num_labels, use_one_hot_embeddings=FLAGS.use_tpu)

        tvars = tf.trainable_variables()

        if FLAGS.init_checkpoint:
            (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars, FLAGS.init_checkpoint)
            tf.train.init_from_checkpoint(FLAGS.init_checkpoint, assignment_map)

        sess_only_input_fro_QA = only_input_fro_QA()

        global_step = tf.get_variable(initializer=0, name="globle_step", trainable=False)

        num_train_steps = int(train_len * num_epoches / train_batch_size)

        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

        # train_op = optimization.create_optimizer(
        #     sess_only_input_fro_QA.loss, learning_rate, num_train_steps, num_warmup_steps, global_step, use_tpu=FLAGS.use_tpu)
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(sess_only_input_fro_QA.loss,global_step=global_step)

        # optimizer = tf.train.AdamOptimizer(learning_rate)
        # gradients, vriables = zip(*optimizer.compute_gradients(sess_bert_lr.total_loss))
        # gradients, _ = tf.clip_by_global_norm(gradients, FLAGS.max_grad_norm)
        # train_op = optimizer.apply_gradients(zip(gradients, vriables), global_step=global_step)
        #
        # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # train_op = tf.group([train_op, update_ops])

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        saver = tf.train.Saver()
        best_auc = 0.0
        best_iter = 0
        early_stop_iter = 1
        for epoch in range(num_epoches):
            elapsed1 = (time.clock() - start1)
            elapsed2 = (datetime.now() - start2)
            print("Time used1:", elapsed1)
            print("Time used2:", elapsed2)
            flag = True
            while flag:
                train_data = sess.run(next_element)

                output_layer1, output_layer2 = sess.run([sess_bert_lr.output_layer1, sess_bert_lr.output_layer2],
                                        feed_dict={sess_bert_lr.input_ids1: train_data["input_ids1"],
                                                   sess_bert_lr.input_mask1: train_data["input_mask1"],
                                                   sess_bert_lr.segment_ids1: train_data["segment_ids1"],
                                                   sess_bert_lr.label_ids1: train_data["label_ids1"],
                                                   sess_bert_lr.is_real_example1: train_data["is_real_example1"],
                                                   sess_bert_lr.input_ids2: train_data["input_ids2"],
                                                   sess_bert_lr.input_mask2: train_data["input_mask2"],
                                                   sess_bert_lr.segment_ids2: train_data["segment_ids2"],
                                                   sess_bert_lr.label_ids2: train_data["label_ids2"],
                                                   sess_bert_lr.is_real_example2: train_data["is_real_example2"],
                                                   sess_bert_lr.is_training: False
                                                   })
                _, loss, acc = sess.run([train_op, sess_only_input_fro_QA.loss, sess_only_input_fro_QA.acc],
                                    feed_dict={
                                        sess_only_input_fro_QA.output_layer1: output_layer1,
                                        sess_only_input_fro_QA.output_layer2: output_layer2,
                                    })


                cur_step = tf.train.global_step(sess, global_step)
                if int(cur_step*train_batch_size/train_len) > epoch:
                    flag = False

                print("epoch:{},global_step:{},loss:{},acc:{}".format(
                    epoch, cur_step, loss, acc))
                if cur_step % save_frequence == 0 and cur_step > start_eval:
                    valid_dev(sess, sess_bert_lr, sess_only_input_fro_QA, data, test_batch_size, eval_len, file="eval")
                    step_auc = valid_dev(sess, sess_bert_lr, sess_only_input_fro_QA, data, test_batch_size, test_len, file="test")
                    if step_auc > best_auc and cur_step >= start_eval:
                        early_stop_iter = 1
                        best_auc = step_auc
                        best_iter = cur_step
                        print('Saving model for step {}'.format(cur_step))
                        saver.save(sess, FLAGS.checkpoint_model_path, global_step=cur_step)
                    elif step_auc < best_auc and cur_step > start_eval:
                        early_stop_iter += 1
                    if early_stop_iter >= early_stop:
                        print("train_over, best_iter={}, best_auc={}".format(best_iter, best_auc))
                        sess.close()
                        exit()


def valid_dev(sess, sess_bert_lr, sess_only_input_fro_QA, data, test_batch_size, test_len, file):
    if file == "eval":
        test_eval = data.creat_eval_dataset()
        label, question_len = reload_data_yahoo.dev_data_label()
    else:
        test_eval = data.creat_test_dataset()
        label, question_len = reload_data_yahoo.test_data_label()
    iterator = test_eval.make_one_shot_iterator()
    next_element = iterator.get_next()

    predict = []

    flag = True
    index = 0
    while flag:
        test_data = sess.run(next_element)
        output_layer1, output_layer2 = sess.run([sess_bert_lr.output_layer1, sess_bert_lr.output_layer2],
                                feed_dict={sess_bert_lr.input_ids1: test_data["input_ids1"],
                                                   sess_bert_lr.input_mask1: test_data["input_mask1"],
                                                   sess_bert_lr.segment_ids1: test_data["segment_ids1"],
                                                   sess_bert_lr.label_ids1: test_data["label_ids1"],
                                                   sess_bert_lr.is_real_example1: test_data["is_real_example1"],
                                                   sess_bert_lr.input_ids2: test_data["input_ids2"],
                                                   sess_bert_lr.input_mask2: test_data["input_mask2"],
                                                   sess_bert_lr.segment_ids2: test_data["segment_ids2"],
                                                   sess_bert_lr.label_ids2: test_data["label_ids2"],
                                                   sess_bert_lr.is_real_example2: test_data["is_real_example2"],
                                                   sess_bert_lr.is_training: False
                                           })

        loss, probability = sess.run([sess_only_input_fro_QA.loss, sess_only_input_fro_QA.score_cand],
                                feed_dict={
                                    sess_only_input_fro_QA.output_layer1: output_layer1,
                                    sess_only_input_fro_QA.output_layer2: output_layer2,
                                })
        index += 1
        if int(index*test_batch_size/test_len) > 0:
            predict.extend(probability[:test_len-((index-1)*test_batch_size)])
            flag = False
        else:
            predict.extend(probability)

    MAP, MRR = new_eval.map_mrr_yahoo(predict, label, question_len)
    print("*" * 30)
    print("{}-MAP:{},{}-MRR:{}".format(file,MAP,file,MRR))
    return MAP


if __name__ == '__main__':
    bert_lr_model(num_epoches=10, shuffle=500, train_batch_size=FLAGS.batch_size, eval_batch_size=FLAGS.batch_size, test_batch_size=FLAGS.batch_size, learning_rate=FLAGS.learning_rate,
                  start_eval=1, save_frequence=100, early_stop=10000, num_labels=2)
