from nets import vgg16
import utils
import tensorflow as tf

img1 = utils.load_image("test_data/table.jpg")

inputs = tf.placeholder(tf.float32, [None, None, 3])
img1 = utils.resize_image(inputs, (224, 224))

prediction = vgg16.vgg_16(img1)

sess = tf.Session()
ckpt_path = "model/vgg_16.ckpt"
sess.run(tf.global_initializer())
saver = tf.train.Saver()
saver.restore(sess, ckpt_path)

probability = tf.nn.softmax(prediction)
pre = sess.run(probability, feed_dict={inputs: img1})

utils.print_prob(probability, './synset.txt')