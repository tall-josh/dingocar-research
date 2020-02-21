import tensorflow as tf 

def test_poop():
    x = tf.constant('poop')
    with tf.Session() as sess:
        actual = sess.run(x)
        actual = actual.decode('utf-8')
        assert actual == 'poop'

