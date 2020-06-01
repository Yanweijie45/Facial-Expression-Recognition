import numpy as np
import os
import tensorflow as tf

channel = 1
default_height = 48
default_width = 48
data_folder_name = '..\data'
data_path_name = 'cv'


# tfrecord数据读出
def __parse_function_csv(serial_exmp_):
    features_ = tf.parse_single_example(serial_exmp_,
                                        features={"image/label": tf.FixedLenFeature([], tf.int64),
                                                  "image/height": tf.FixedLenFeature([], tf.int64),
                                                  "image/width": tf.FixedLenFeature([], tf.int64),
                                                  "image/raw": tf.FixedLenFeature([default_width*default_height*channel]
                                                                                  , tf.int64)})
    label_ = tf.cast(features_["image/label"], tf.int32)
    height_ = tf.cast(features_["image/height"], tf.int32)
    width_ = tf.cast(features_["image/width"], tf.int32)
    image_ = tf.cast(features_["image/raw"], tf.int32)
    image_ = tf.reshape(image_, [height_, width_, channel])
    image_ = tf.multiply(tf.cast(image_, tf.float32), 1. / 255)
    image_ = tf.image.random_flip_left_right(image_)
    image_ = tf.image.random_brightness(image_, max_delta=32. / 255)
    image_ = tf.image.random_contrast(image_, lower=0.8, upper=1.2)
    image_ = tf.random_crop(image_,
                            [default_height - np.random.randint(0, 4), default_width - np.random.randint(0, 4), 1])
    image_ = tf.image.resize_images(image_, [default_height, default_width])
    return image_, label_


#获取数据集
def get_dataset(record_name_):
    record_path_ = os.path.join(data_folder_name, data_path_name, record_name_)
    data_set_ = tf.data.TFRecordDataset(record_path_)
    return data_set_.map(__parse_function_csv)
