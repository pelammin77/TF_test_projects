import tensorflow as tf

node1 = tf.constant(2.0, tf.float32)
node2 = tf.constant(5.0, tf.float32)

res = node1 + node2

# print(res) # prints: Tensor("add:0", shape=(), dtype=float32) not res value

sess = tf.Session()
file_writer = tf.summary.FileWriter('graph', sess.graph) # makes the graph
print(sess.run(res))
sess.close()  # close the session