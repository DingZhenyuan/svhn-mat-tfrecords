import tensorflow as tf
from scipy.io import loadmat


# 从路径中提取image和label
def data_set(name, num_sample_size=10000):
    if name == 'train':
        filename = "svhn_data/format2/train_32x32.mat"
    elif name == 'test':
        filename = "svhn_data/format2/test_32x32.mat"
    elif name == 'extra':
        filename = "svhn_data/format2/extra_32x32.mat"
    else:
        print("The name is wrong!")
    datadict = loadmat(filename)
    image = datadict['X']
    image = image.transpose((3, 0, 1, 2))

    label = datadict['y'].flatten()
    label[label == 10] = 0  # 修正10--1

    # 分割时用
    # image = image[:num_sample_size]
    # label = label[:num_sample_size]
    return image, label


# 封装数据格式
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# 封装数据格式
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to_tfrecords(images, labels, fileName):
    num_examples, rows, cols, depth = images.shape
    print(num_examples)

    # 写入过程
    print('Writing', fileName)
    writer = tf.python_io.TFRecordWriter(fileName)
    for index in range(num_examples):
        image_raw = images[index].tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(rows),
            'width': _int64_feature(cols),
            'depth': _int64_feature(depth),
            'label': _int64_feature(int(labels[index])),
            'image_raw': _bytes_feature(image_raw)}))
        writer.write(example.SerializeToString())
    writer.close()


# 拆分数据集
def split_dataset(train_x, train_y, validation_size):
    return (train_x[:-validation_size],
            train_y[:-validation_size],
            train_x[-validation_size:],
            train_y[-validation_size:])


with tf.Session() as sess:
    train_x, train_y = data_set('train', 73257)
    test_x, test_y = data_set('test', 26032)
    extra_x, extra_y = data_set('extra', 10000)

    # 需要分割获得validation时用
    # train_x, train_y, valid_x, valid_y = split_dataset(train_x, train_y, 1000)

    trainFileName = "svhn_data/format2/train.tfrecords"
    testFileName = "svhn_data/format2/test.tfrecords"
    extraFileName = "svhn_data/format2/extra.tfrecords"

    # Convert to Examples and write the result to TFRecords.
    convert_to_tfrecords(train_x, train_y, trainFileName)
    convert_to_tfrecords(test_x, test_y, testFileName)
    convert_to_tfrecords(extra_x, extra_y, extraFileName)
    print('Over')