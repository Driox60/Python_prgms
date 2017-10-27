# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 14:39:10 2017

@author: AT522
"""

##############################
# Utilisation de Tensorboard #
##############################

import tensorflow as tf
sess = tf.Session()

# defs des constantes, placeholders et fonctions
# les "name" servent à tensorboard
node1 = tf.constant(3.0, name="Node1_constante_3.0")
node2 = tf.constant(4.0, name="Node2_constante_4.0")
node3 = tf.add(node1, node2, name="Add_node1_node2")

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

adder_node = tf.add(a, b, name="adder_node_add_placeholders_a_et_b")
add_and_triple = tf.multiply(adder_node, 3, name="multiply_adder_node_by_3")

# We add "SummaryWriter", this will create a folder which will contain the
# information for TensorBoard to build the graph. 
writer = tf.summary.FileWriter("D:\\tmp\\tensorflow\\logs", sess.graph)

print("node1 (fixed constante) : ", sess.run(node1))
print("node2 (fixed constante) : ", sess.run(node2))
print("node3 (node1 + node2) : ", sess.run(node3))
print ("adder_node (via inputs 3 & 4.5 from Placeholder) :", sess.run(adder_node, {a: 3, b: 4.5}))
print ("add_and_triple (via inputs from Placeholder) :", sess.run(add_and_triple, {a: 3, b: 4.5}))

tensorboard --logdir="D:\\tmp\\tensorflow\\logs"

writer.close()