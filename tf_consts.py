import tensorflow as tf

node1 = tf.constant(2.0, tf.float32)
node2 = tf.constant(5.0, tf.float32)

res = node1 + node2

# print(res) # prints: Tensor("add:0", shape=(), dtype=float32) not res value

sess = tf.Session()
print(sess.run(res))
sess.close()  # close the session

# or you can use with sentence

with tf.Session() as sess:
    output = sess.run([res])
    print("with sentence", output)
    # you don't need to close session if you use 'with'

