from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import time
import sys
import tensorflow.compat.v1 as tf
import tensorflow as tf
import argparse

# tf.compat.v1.disable_eager_execution()
# tf.compat.v1.Session()
tf.keras.backend.set_floatx('float64')
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="mnist")
parser.add_argument("--arch", default="1blk_64")
parser.add_argument("--relu", type=int, default=0)  # default=0: relu without upper bound
parser.add_argument("--epoch", type=int, default=1)  # default=0: relu without upper bound
parser.add_argument("--qu_bit", type=int, default=6)  # default=0: relu without upper bound
# parser.add_argument("--resize", type=int, default=10)

args = parser.parse_args()

arch = args.arch.split('_')
numBlk = arch[0][:-3]
blkset = list(map(int, arch[1:]))  # without output layer

# blkset.append(10)

if args.dataset == "fashion-mnist":
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
elif args.dataset == "mnist":
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
elif args.dataset == 'svhn':
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.svhn.load_data()
else:
    raise ValueError("Unknown dataset '{}'".format(args.dataset))

effective_dim = np.prod(x_train[0].shape[1:])
x_train = x_train.reshape(x_train.shape[0], effective_dim)
x_test = x_test.reshape(x_test.shape[0], effective_dim)

x_train /= 255
x_test /= 255

model = Sequential()
in_input_dim = effective_dim

index = 0
for i in blkset:
    model.add(tf.keras.layers.Dense(i, activation='relu', name='Dense_' + str(index), input_dim=in_input_dim))
    index += 1
    in_input_dim = i

model.add(tf.keras.layers.Dense(10))

model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

train_begin_time = time.time()

model.fit(x_train, y_train, epochs=args.epoch, validation_data=(x_test, y_test))
train_end_time = time.time()

training_time = train_end_time - train_begin_time

loss, accu = model.evaluate(x_test, y_test)

print("Loss is: ", loss)
print("Accuracy is: ", accu)

model_path = "benchmark_Quadapter/" + args.dataset + str(args.arch) + "_model.h5"
weight_path = "benchmark_Quadapter/" + args.datasetstr(args.arch) + "_weight.h5"
model.save(model_path)
model.save_weights(weight_path)

logPath = "benchmark_Quadapter/" + args.dataset + str(args.arch) + "_accuracy.txt"
fo = open(logPath, "w")
fo.write("Loss: " + str(loss) + "\n")
fo.write("sparse_categorical_accuracy: " + str(accu) + "\n")
fo.write("training time: " + str(training_time))
fo.close()
