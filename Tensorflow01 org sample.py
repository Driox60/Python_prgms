# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 14:39:10 2017

@author: AT522
"""

########################################################
# programmes et exemples venant du site tensorflow.org #
########################################################

import tensorflow as tf

# tensorflow est composé de 2 sections : 1) build computational graph, and 
# 2) run computational graph. Computational graph = série d'opérations tensorflow
# arrangées en nodes. 

# un type de node peut être une constante. Ici on crée 2 constantes 3.0 & 4.0   
node1 = tf.constant(3.0)
node2 = tf.constant(4.0)

# l'impression des constantes ne donne pas leur valeur. pour "évaluer" ces nodes
# et obtenir en sortie 3.0 & 4.0 , il faut faire tourner le computational graph
# dans une session    
print ("----------------------")
print ("node1 = ", node1)
print ("node2 = ", node2)


# On defini une session, et on obtient bien les valeurs 3.0 et 4.0 
sess = tf.Session()
print ("affichage de node1 et node2 dans une session tensorflow :")
print(sess.run([node1, node2]))
print ("---------------------")
# On peut combiner les nodes avec des operations (qui sont alors aussi des nodes)
node3 = tf.add(node1, node2)
print ("node3 = ", node3)
print ("affichage de node3 dans une session tensorflow :")
print("sess.run(node3)", sess.run([node3]))
print ("---------------------")

# Pas très interessant si on utilise que des constantes.... Un graph peut être
# paramétré pour accepter aussi des entrées extérieures, que l'on met dans un 
# placeholder. Un placeholder est un endroit qui fournira une valeur plus tard.

# on défini une fonction avec 2 parametres a et b et une operation (+)
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = tf.add(a, b)

# on évalue ce graph en fournissant des entrées (feed-dict argument) aux  
# placeholders lorsque l'on lance la session. 
print ("affichage fonction adder_node") 
print (sess.run(adder_node, {a: 3, b: 4.5}))
print (sess.run(adder_node, {a: [1, 3, 7], b: [2, 4, 7]}))

# on peut même ensuite ajouter d'autres operations 
print ("affichage fonction add_and_triple")
add_and_triple = adder_node * 3
print (sess.run(add_and_triple, {a: 3, b: 4.5}))
print ("---------------------")

# Pour que nos modèles puissent s'entrainer, il faut pouvoir modifier le graph 
# afin d'obtenir de nouvelles outputs avec les mêmes inputs. "Variable" permet
# d'ajouter des paramètres d'entrainement à un graph. "Variable" est construit
# avec un type et une valeur initiale : 
print (" ")
print ("--------------------------------------------")
print ("affichage fonction modele_lineaire = a x + b")
print ("--------------------------------------------")
a = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
modele_lineaire = a * x + b 

# Pour initialiser les variables en Tensorflow, on doit explicitement appeler
# la cmde init suivante :
init = tf.global_variables_initializer()
sess.run(init)

# comme x est un placeholder, on peut évaluer notre modèles avec plusieurs
# valeurs de x simultanément : 
print(sess.run(modele_lineaire, {x: [1, 2, 3, 4]})) 

# On a donc créé un modèle, mais on ne sait pas si il est bon ou pas. Pour
# l'évaluer avec des données d'entrainement, on a besoin d'un placeholder "y" en +
# qui va fournir les valeurs désirées , et on va écrire une "loss function" .
# La "loss function" va mesurer la distance entre le modèle et les données 
# fournies. On va utiliser un modele standard de régression linéaire, en faisant
# la somme des carrés des deltas entre le modèle et les données fournies.  
# "modele_lineaire - y " créé un vecteur ou chaque élément est l'erreur delta
# de l'exemple correspondant. 
# "tf.square" prend le carré de l'erreur.
# Ensuite on somme tous les carrés pour obtenir une simple valeur qui résume toutes     
# les erreurs des exemples, en utilisant "tf.reduce.sum" : 
    
y = tf.placeholder(tf.float32)
carre_des_deltas = tf.square(modele_lineaire - y)
loss = tf.reduce_sum(carre_des_deltas)
print(" ")
print("resultat fonction loss avec les valeurs initiales .3 et -.3 : ")
print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))    

# on obtient 23.66 . On peut l'améliorer manuellement et visuellement, car on
# voit que si on choisit a= -1 et b = 1 , ce sont les paramètres optimaux pour 
# notre modèle, et on obtient 0.0 :

fixa = tf.assign(a, [-1.])
fixb = tf.assign(b, [1.])
sess.run([fixa, fixb])
print("resultat fonction loss avec valeurs changées à -1. et 1. :")
print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))  

# on a deviné les valeurs parfaites, mais le but du Machine Learning est de 
# trouver ces valeurs automatiquement. Pour ceci tensorflow fournit des 
# "optimizers" qui vont changer lentement les valeurs des variables afin de
# minimiser la fonction loss. L'optimizer le + simple est "gradient descent".
# il modifie chaque variable en fonction de l'amplitude de la dérivée de la perte
# par rapport à cette variable. 
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# données d'entrainement 
x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]

# boucle d'entrainement
sess.run(init)           # reset values to "incorrect" defaults
for i in range (1000):
    sess.run(train, {x: x_train, y: y_train})

# evaluation de l'acuracy (exactitude) de l'entrainement
print (" ")
print ("------------------------------------------------")
print ("resultats avec valeurs trouvées par Tensorflow :") 
print(sess.run([a, b]))
curr_a, curr_b, curr_loss = sess.run([a, b, loss], {x: x_train, y:y_train})
print ("-----------------------------------------------")
print ("a: ", curr_a, "b: ", curr_b, "loss: ", curr_loss)
    
    