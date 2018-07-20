import tensorflow as tf
import numpy as np

a=np.asarray([[0.1,0.3,0.6]])
print(a.shape)
ph = tf.placeholder(tf.float32,(1,3))
print(ph.shape)

multi = tf.distributions.Multinomial([1.], probs=ph)

sample = multi.sample()

print(sample.shape)
sess=tf.Session()

for i in range(10):
    rslt = sess.run(sample, feed_dict={ph:a})
    print (rslt)
    print (np.argmax(rslt[0]))
