import tensorflow as tf
import numpy as np

a=np.asarray([[1,2,3]])
print(a.shape)
ph = tf.placeholder(tf.float32,(1,3))
print(ph.shape)

multi = tf.multinomial(ph,12)

sess=tf.Session()

rslt = sess.run(multi, feed_dict={ph:a})

print (rslt.shape)
print (rslt)
