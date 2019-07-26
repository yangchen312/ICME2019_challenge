import numpy as np
import tensorflow as tf
from functools import reduce


class PosShifts(object):
  """
  Position shifts will be caused by different fields,
  thus, we need to take it into our consideration.
  This class is used for removing position shifts
  """

  _shifts = []
  def __init__(self, track):
    PosShifts._track = track
    if track == 1:
      PosShifts._shifts = [663011, 0, 31180492, 15595718, 410, 6]
    elif track == 2:
      PosShifts._shifts = [73974, 396, 4122689, 850308, 461, 5, 89778, 75085]
    else:
      raise Exception("unknown track", track)

  @staticmethod
  def get_features_num():
    index_shift = PosShifts._shifts
    all_shift = reduce(lambda x, y: x+y, index_shift)
    return all_shift

  @staticmethod
  def shift():
    """ position shifts for different field features """
    shifts = PosShifts._shifts
    shifts = [0] + shifts

    sum = 0
    for index, shift in enumerate(shifts):
      sum += shift
      shifts[index] = sum
    return shifts


class LineParser(object):
  """
  class for parsing tf input line
  """
  @staticmethod
  def parse_finish_line(line):
    """
    tf operator not good for parse text info
    thus py_func is applied
    finish line parse
    """
    return tf.py_func(DataParser.data_parser, [line, 6], [tf.int32, tf.float32, tf.float32])

  @staticmethod
  def parse_like_line(line):
    """
    tf operator not good for parse text info
    thus py_func is applied
    like line parser
    """
    return tf.py_func(DataParser.data_parser, [line, 7], [tf.int32, tf.float32, tf.float32])


class DataParser(object):
  """
  Detailed operator foe line input
  """

  @staticmethod
  def data_parser(line, label_index):
    """ parser line content and generate idx, features, and gts """
    content = line.split(b'\t')
    label = np.float32(content[label_index].strip())
    features = [content[i] for i in [0, 1, 2, 3, 4]]
    idx = [0 if feature < 0 else feature for feature in map(lambda feature: np.float32(feature), features)]
    features = [np.float32(0) if feature < 0 else np.float32(1) for feature in map(lambda feature: np.float32(feature), features)]

    shifts = PosShifts.shift()
    idx = [idx[i] + shifts[i] for i in range(len(idx))]
    idx = list(map(lambda one_id: np.int32(one_id), idx))

    return idx, features, label

