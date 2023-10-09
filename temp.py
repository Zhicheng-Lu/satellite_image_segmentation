import tensorflow as tf

dummy_input = tf.random_normal([3], mean=0, stddev=1)
dummy_input = tf.Print(dummy_input, data=[dummy_input],
                           message='New dummy inputs have been created: ', summarize=6)
q = tf.FIFOQueue(capacity=3, dtypes=tf.float32)
enqueue_op = q.enqueue_many(dummy_input)








data = q.dequeue()
# data = tf.Print(data, data=[data], message='This is how many items are left in q: ')
# create a fake graph that we can call upon
# fg = data + 1

with tf.Session() as sess:
    # first load up the queue
    sess.run(enqueue_op)
    # now dequeue a few times, and we should see the number of items
    # in the queue decrease
    x = sess.run(data)
    print(x)
    sess.run(data)
    sess.run(data)
    # by this stage the queue will be emtpy, if we run the next time, the queue
    # will block waiting for new data
    # sess.run(fg)
    # this will never print:
    print("We're here!")
