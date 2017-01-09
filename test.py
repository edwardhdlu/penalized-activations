import tensorflow as tf
import numpy as np
penalized_activations = tf.load_op_library("penalized_activations.so")

with tf.Session(""):
	arr = np.array([[ -2.2, -1, 0, 2, 1000.101 ]])
	r1 = penalized_activations.penalized_tanh(arr, 0.25).eval()
	r2 = penalized_activations.penalized_tanh(arr, 1.0).eval()
	r3 = tf.tanh(arr).eval()

	r4 = penalized_activations.leaky_relu(arr, 0.25).eval()
	r5 = penalized_activations.leaky_relu(arr, 1.0).eval()
	r6 = tf.nn.relu(arr).eval()

	print r1
	print r2
	print r3

	print r4
	print r5
	print r6
