import tensorflow as tf

W = tf.Variable([0.3], tf.float32)
b = tf.Variable([-0.3], tf.float32)
x = tf.placeholder(tf.float32)
model = W * x + b
y = tf.placeholder(tf.float32)

# calculate loss
squared_delta = tf.square(model - y)
loss = tf.reduce_sum(squared_delta)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
sess.run(init)
# W * x + b = y
print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))  # loss is 23.66
"""
 if W is -1 and b is 1 loss is 0 

    W * x + b = y 
    -1 * 1 + 1 = 0
    -1* 2 + 1 = -1
    -1 * 3 + 1 = -2 
    -1 * 4 + 1 = -3

"""
