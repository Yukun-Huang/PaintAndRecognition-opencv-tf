import sys
import tensorflow as tf
from PIL import Image, ImageFilter

sess = tf.Session()
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
saver = tf.train.Saver()
saver.restore(sess, "model1/model.ckpt")
print "Session set,load linear model success!"

def predictint(imvalue):
    x = tf.placeholder(tf.float32, [1, 784])
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    # init_op = tf.global_variables_initializer()
    # sess.run(init_op)
    prediction=tf.argmax(y,1)
    predint = prediction.eval(feed_dict={x: [imvalue]}, session=sess)
    print predint
    return (predint[0],)
