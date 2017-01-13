import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import gen_math_ops
import numpy as np
import pickle

# tf.nn.relu <---> penalized_activations.leaky_relu
penalized_activations = tf.load_op_library("penalized_activations.so")
@ops.RegisterGradient("LeakyRelu")
def _leaky_relu_grad(op, grad):
	x = op.inputs[0]
	a = op.inputs[1]

	return tf.select(x < 0., gen_nn_ops._relu_grad(grad, x * a), gen_nn_ops._relu_grad(grad, x)), tf.zeros(tf.shape(a))

@ops.RegisterGradient("PenalizedTanh")
def _penalized_tanh_grad(op, grad):
	x = op.inputs[0]
	a = op.inputs[1]

	return tf.select(x < 0., gen_math_ops._tanh_grad(grad, x * a), gen_math_ops._tanh_grad(grad, x)), tf.zeros(tf.shape(a))

# setup
train_file = file("data/dataset_train_10k.csv")
val_file = file("data/dataset_val_10k.csv")
test_file = file("data/dataset_test_10k.csv")

batch_size = 50
epochs = 100
num_labels = 10

# converts an array to onehot
def onehot(label, length):
	arr = [0] * length
	arr[label] = 1
	return arr

# read in input file as np arrays
def parse(input_file):
	labels = []
	inputs = []

	for line in input_file:
		row = line.split(",")

		labels.append(int(row[0]))
		inputs.append([float(x) - 0.5 for x in row[1:]]) # regularized

	np_labels_pre = []
	np_inputs = np.array(inputs).astype(np.float32)

	for label in labels:
		label = 0 if label == 10 else label # tweak for SHVN labelling
		np_labels_pre.append(onehot(label, num_labels))

	np_labels = np.matrix(np_labels_pre).astype(np.uint8)

	return np_inputs, np_labels

# convolution
def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")

# max pooling
def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

# read in input
print "Reading in files..."
train_data, train_labels = parse(train_file);
test_data, test_labels = parse(test_file);
val_data, val_labels = parse(val_file)
print "Training..."

train_size, num_features = train_data.shape
image_size = int(num_features ** 0.5)

# define the model
x = tf.placeholder(tf.float32, shape=[None, 1024])
y_ = tf.placeholder(tf.float32, shape=[None, num_labels])

# first layer
W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1))
b_conv1 = tf.Variable(tf.zeros([32]))

x_image = tf.reshape(x, [-1, image_size, image_size, 1])

h_conv1 = penalized_activations.leaky_relu(conv2d(x_image, W_conv1) + b_conv1, 0.25)
h_pool1 = max_pool_2x2(h_conv1)

# second layer
W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
b_conv2 = tf.Variable(tf.zeros([64]))

h_conv2 = penalized_activations.leaky_relu(conv2d(h_pool1, W_conv2) + b_conv2, 0.25)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = tf.Variable(tf.truncated_normal([8 * 8 * 64, 1024], stddev=0.1))
b_fc1 = tf.Variable(tf.zeros([1024]))

h_pool2_flat = tf.reshape(h_pool2, [-1, 8 * 8 * 64])
h_fc1 = penalized_activations.leaky_relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1, 0.25)

# dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = tf.Variable(tf.truncated_normal([1024, num_labels], stddev=0.1))
b_fc2 = tf.Variable(tf.zeros([num_labels]))

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver([W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_fc2, b_fc2])

# run training
with tf.Session() as session:
	tf.initialize_all_variables().run()
	for step in range(epochs):
		offset = 0
		for i in range(train_size / batch_size):
			batch_data = train_data[offset:offset + batch_size]
			batch_labels = train_labels[offset:offset + batch_size]

			train_step.run(feed_dict={x: batch_data, y_: batch_labels, keep_prob: 0.75})
			offset += batch_size

		train_accuracy = accuracy.eval(feed_dict={x: val_data, y_: val_labels, keep_prob: 1.0})
		print "Step " + str(step) + "/" + str(epochs) + " accuracy: " + str(train_accuracy)

	print "Final accuracy: " + str(accuracy.eval(feed_dict={x: test_data, y_: test_labels, keep_prob: 1.0}))

	print "Saving model..."
	saver.save(session, "model")
	