# -*- coding: utf-8 -*-

import tensorflow as tf
sess = tf.Session()

# parms
a = tf.Variable([2.0], dtype=tf.float32, name="a")
x = tf.placeholder(tf.float32, name="x")
b = tf.Variable([1.0], dtype=tf.float32, name="b")

# model : y=ax+b
with tf.name_scope('Model'):
    y = tf.add ((tf.multiply(a, x)), b)

# info for TensorBoard 
writer = tf.summary.FileWriter("D:\\tmp\\tensorflow\\logs", sess.graph)

# loss fct - mean square error
with tf.name_scope('cost'):
    y_prim = tf.placeholder(tf.float32)
    cost = tf.reduce_sum(tf.square(y - y_prim))

# optimizer = gradientdescent
with tf.name_scope('GradDes'):
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(cost)

# train datas 
x_train = [1, 2, 3, 4]
y_train = [5.2, 8.4, 11.1, 14.7]

# Create a summary to monitor cost tensor
tf.summary.scalar("cost", cost)
# Merge all summaries into a single op
merged_summary_op = tf.summary.merge_all()

# init vars
init = tf.global_variables_initializer()

# train loop
sess.run(init)  
for i in range (500):
    sess.run([train, cost], feed_dict={x: x_train, y_prim: y_train})
    a_found, b_found, curr_cost, summary = sess.run([a, b, cost, merged_summary_op], feed_dict={x: x_train, y_prim: y_train}) 
    writer.add_summary(summary)
    writer.flush()
    print("iteration :", i, "a: ", a_found, "b: ", b_found, "cost: ", curr_cost)
    

writer.close()

# A lancer en command prompt dos : tensorboard --logdir D:\tmp\tensorflow\logs
# on peut aussi lancer : tensorboard --logdir D:\tmp\tensorflow\logs --purge_orphaned_data
# faire tensorboard --hlp pour voir ttes les options possibles  
# A partir de Chrome (de préférence) : http://AUDIT-PC:6006
# il vaur mieux disabler avast le tps d'utiliser tensorboard, et se placer ds la directory adequate pour lancer la cmd.




