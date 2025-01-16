import model
import utils
import tensorflow as tf

image1 = utils.load_image('table.jpg')
input = tf.placeholder(tf.float32, [None, 224, 224, 3])
resized_image = tf.image.resize(input, (224, 224))
prediction = model.vgg_16(resized_image)
sess = tf.Session()
ckpt = 'vgg_16.ckpt'
init = tf.global_variables_initializer()
sess.run(init)
saver = tf.train.Saver()
saver.restore(sess, ckpt)
pro = tf.nn.softmax(prediction)
print(sess.run(pro, feed_dict={input: image1}))
