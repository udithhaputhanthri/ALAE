import matplotlib
import tensorflow as tf
import os
from matplotlib import image

def save_test_images(images):
    os.mkdir('mnist_test_imgs')
    i=0
    for img in images:
        i+=1
        image.imsave(f'mnist_test_imgs/{i}.png', img)  

mnist = tf.keras.datasets.mnist.load_data(path="mnist.npz")
mnist_train, mnist_test = mnist[0], mnist[1]
save_test_images(mnist_test[0])
