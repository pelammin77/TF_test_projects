import tensorflow as tf

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b
sess = tf.Session()

result_node = sess.run(adder_node, {a: [1, 2], b: [2, 4]})
print('result:', result_node)
sess.close()
