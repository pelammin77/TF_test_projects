import tensorflow as tf

W = tf.Variable([-0.3], tf.float32)
b = tf.Variable([0.3], tf.float32)
x = tf.placeholder(tf.float32)
model = W * x + b
y = tf.placeholder(tf.float32)

# calculate loss
squared_delta = tf.square(model - y)
loss = tf.reduce_sum(squared_delta)

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)


init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(10000):
    sess.run(train,{x: [1, 2, 3, 4], y: [0, -1, -2, -3]})
    if i % 1000 == 0:
        print(sess.run([W, b]))



print(sess.run([W, b])) # W = -0.99999911 b =  0.99999744 very close to  -1 and 1
