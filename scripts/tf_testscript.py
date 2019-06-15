import numpy as np
import tensorflow as tf


x = np.random.uniform(-100, 100, (10000, 40))
y = np.sin(x)


# Initialize two constants
x1 = tf.constant([1, 2, 3, 4])
x2 = tf.constant([5, 6, 7, 8])

# Multiply
result = tf.multiply(x1, x2)

with tf.device("/gpu:2"):
	session = tf.Session()

	np.savetxt("testfile.csv",session.run(result))

	session.close()
