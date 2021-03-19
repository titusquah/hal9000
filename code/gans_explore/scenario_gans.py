import scipy.misc
import numpy as np
from numpy import shape
import tensorflow as tf
import os
import sys
import csv
def load_wind_data_new():
    #data created on Oct 3rd, WA 20 wind farms, 7 years
    with open('real.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        rows = [row for row in reader]
    rows = np.array(rows[0:736128], dtype=float)
    trX = []
    print(shape(rows))
    m = np.ndarray.max(rows)
    print("Maximum value of wind", m)
    print(shape(rows))
    for x in range(20):
        train = rows[:736128, x].reshape(-1, 576)
        train = train / 16

        # print(shape(train))
        if trX == []:
            trX = train
        else:
            trX = np.concatenate((trX, train), axis=0)
    print("Shape TrX", shape(trX))

    with open('sample_label.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        rows = [row for row in reader]
    trY = np.array(rows, dtype=int)
    print("Label Y shape", shape(trY))

    with open('index.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        index = [row for row in reader]
    index=np.array(index, dtype=int)

    print(shape(index))
    print('hello world!')

    trX2=trX[index[0:23560]]
    trY2=trY[index[0:23560]]
    trX2=trX2.reshape([-1,576])
    teX=trX[index[23560:]]
    teX = teX.reshape([-1, 576])
    teY=trY[index[23560:]]

    with open('trainingX.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        samples = np.array(trX2*16, dtype=float)
        writer.writerows(samples.reshape([-1, 576]))

    with open('trainingY.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        samples = np.array(trY2, dtype=float)
        writer.writerows(samples)

    with open('testingX.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        samples = np.array(teX*16, dtype=float)
        writer.writerows(samples.reshape([-1, 576]))

    with open('testingY.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        samples = np.array(teY, dtype=float)
        writer.writerows(samples)

    with open('24_hour_ahead_full.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        rows = [row for row in reader]
    rows = np.array(rows[0:736128], dtype=float)
    forecastX = []
    print(shape(rows))
    m = np.ndarray.max(rows)
    m=np.clip(m,0, 16.0)
    print("Maximum value of wind", m)
    print(shape(rows))
    for x in range(20):
        train = rows[:736128, x].reshape(-1, 576)
        train = train / 16

        # print(shape(train))
        if forecastX == []:
            forecastX = train
        else:
            forecastX = np.concatenate((forecastX, train), axis=0)
    print("Shape ForecastX", shape(forecastX))
    forecastX=forecastX[index[23560:]]
    forecastX = forecastX.reshape([-1, 576])
    return trX2, trY2, teX, teY, forecastX


def batchnormalize(X, eps=1e-8, g=None, b=None):
    if X.get_shape().ndims == 4:
        mean = tf.reduce_mean(X, [0,1,2])
        std = tf.reduce_mean( tf.square(X-mean), [0,1,2] )
        X = (X-mean) / tf.sqrt(std+eps)

        if g is not None and b is not None:
            g = tf.reshape(g, [1,1,1,-1])
            b = tf.reshape(b, [1,1,1,-1])
            X = X*g + b

    elif X.get_shape().ndims == 2:
        mean = tf.reduce_mean(X, 0)
        std = tf.reduce_mean(tf.square(X-mean), 0)
        X = (X-mean) / tf.sqrt(std+eps)

        if g is not None and b is not None:
            g = tf.reshape(g, [1,-1])
            b = tf.reshape(b, [1,-1])
            X = X*g + b

    else:
        raise NotImplementedError

    return X

def lrelu(X, leak=0.2):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * X + f2 * tf.abs(X)

def bce(o, t):
    o = tf.clip_by_value(o, 1e-7, 1. - 1e-7)
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=o, logits=t))

class DCGAN():
    def __init__(
            self,
            batch_size=100,
            image_shape=[24,24,1],
            dim_z=100,
            dim_y=5,
            dim_W1=1024,
            dim_W2=128,
            dim_W3=64,
            dim_channel=1,
            lam=0.05
            ):

        self.lam=lam
        self.batch_size = batch_size
        self.image_shape = image_shape
        self.dim_z = dim_z
        self.dim_y = dim_y

        self.dim_W1 = dim_W1
        self.dim_W2 = dim_W2
        self.dim_W3 = dim_W3
        self.dim_channel = dim_channel

        self.gen_W1 = tf.Variable(tf.random_normal([dim_z+dim_y, dim_W1], stddev=0.02), name='gen_W1')
        self.gen_W2 = tf.Variable(tf.random_normal([dim_W1+dim_y, dim_W2*6*6], stddev=0.02), name='gen_W2')
        self.gen_W3 = tf.Variable(tf.random_normal([5,5,dim_W3,dim_W2+dim_y], stddev=0.02), name='gen_W3')
        self.gen_W4 = tf.Variable(tf.random_normal([5,5,dim_channel,dim_W3+dim_y], stddev=0.02), name='gen_W4')

        self.discrim_W1 = tf.Variable(tf.random_normal([5,5,dim_channel+dim_y,dim_W3], stddev=0.02), name='discrim_W1')
        self.discrim_W2 = tf.Variable(tf.random_normal([5,5,dim_W3+dim_y,dim_W2], stddev=0.02), name='discrim_W2')
        self.discrim_W3 = tf.Variable(tf.random_normal([dim_W2*6*6+dim_y,dim_W1], stddev=0.02), name='discrim_W3')
        self.discrim_W4 = tf.Variable(tf.random_normal([dim_W1+dim_y,1], stddev=0.02), name='discrim_W4')



    def build_model(self):
        Z = tf.placeholder(tf.float32, [self.batch_size, self.dim_z])
        Y = tf.placeholder(tf.float32, [self.batch_size, self.dim_y])

        image_real = tf.placeholder(tf.float32, [self.batch_size]+self.image_shape)
        pred_high = tf.placeholder(tf.float32, [self.batch_size]+self.image_shape)
        pred_low = tf.placeholder(tf.float32, [self.batch_size]+self.image_shape)
        h4 = self.generate(Z, Y)
        #image_gen comes from sigmoid output of generator
        image_gen = tf.nn.sigmoid(h4)

        raw_real2 = self.discriminate(image_real, Y)
        #p_real = tf.nn.sigmoid(raw_real)
        p_real = tf.reduce_mean(raw_real2)

        raw_gen2 = self.discriminate(image_gen, Y)
        #p_gen = tf.nn.sigmoid(raw_gen)
        p_gen = tf.reduce_mean(raw_gen2)


        discrim_cost = tf.reduce_mean(raw_real2) - tf.reduce_mean(raw_gen2)
        gen_cost = -tf.reduce_mean(raw_gen2)

        mask = tf.placeholder(tf.float32, [self.batch_size] + self.image_shape, name='mask')
        '''contextual_loss_latter = tf.reduce_sum(tf.contrib.layers.flatten(
            -tf.log(tf.abs(image_real-image_gen))), 1)'''
        #contextual_loss_latter = tf.reduce_sum(tf.log(tf.contrib.layers.flatten(tf.abs(image_gen - pred_high))), 1)

        #log loss
        '''contextual_loss_latter = tf.reduce_sum(tf.contrib.layers.flatten(
        -tf.log(tf.maximum(
            (mask + tf.multiply(tf.ones_like(mask) - mask, pred_high)) - tf.multiply(
                tf.ones_like(mask) - mask, image_gen), 0.0001*tf.ones_like(mask)))
        -tf.log(tf.maximum(
            (mask + tf.multiply(tf.ones_like(mask) - mask, image_gen)) - tf.multiply(
                tf.ones_like(mask) - mask, pred_low), 0.0001*tf.ones_like(mask)))), 1)'''
        contextual_loss_latter = tf.contrib.layers.flatten(
            -tf.log(
                (mask + tf.multiply(tf.ones_like(mask) - mask, pred_high)) - tf.multiply(
                    tf.ones_like(mask) - mask, image_gen))
            - tf.log(
                (mask + tf.multiply(tf.ones_like(mask) - mask, image_gen)) - tf.multiply(
                    tf.ones_like(mask) - mask, pred_low)))
        contextual_loss_latter = tf.where(tf.is_nan(contextual_loss_latter), tf.ones_like(contextual_loss_latter) * 1000000.0, contextual_loss_latter)
        contextual_loss_latter2 = tf.reduce_sum(contextual_loss_latter, 1)
        #square loss
        '''contextual_loss_latter = tf.reduce_sum(tf.contrib.layers.flatten(
            tf.square(tf.multiply(tf.ones_like(mask) - mask, image_gen) - tf.multiply(tf.ones_like(mask) - mask, pred_high)))
        +tf.contrib.layers.flatten(
            tf.square(
                tf.multiply(tf.ones_like(mask) - mask, image_gen) - tf.multiply(tf.ones_like(mask) - mask, pred_high)))
        , 1)'''
        contextual_loss_former = tf.reduce_sum(tf.contrib.layers.flatten(
            tf.square(tf.multiply(mask, image_gen) - tf.multiply(mask, image_real))), 1)
        contextual_loss_prepare = tf.reduce_sum(tf.contrib.layers.flatten(
            tf.square(tf.multiply(tf.ones_like(mask) - mask, image_gen) - tf.multiply(tf.ones_like(mask)-mask, image_real))), 1)
        perceptual_loss = gen_cost
        complete_loss = contextual_loss_former + self.lam * perceptual_loss + 0.05*contextual_loss_latter2
        grad_complete_loss = tf.gradients(complete_loss, Z)
        grad_uniform_loss = tf.gradients(contextual_loss_prepare, Z)

        return Z, Y, image_real, discrim_cost, gen_cost, p_real, p_gen, grad_complete_loss, \
               pred_high, pred_low, mask, contextual_loss_latter, contextual_loss_former, grad_uniform_loss


    def discriminate(self, image, Y):
        yb = tf.reshape(Y, tf.stack([self.batch_size, 1, 1, self.dim_y]))
        X = tf.concat([image, yb * tf.ones([self.batch_size, 24, 24, self.dim_y])],3)

        h1 = lrelu( tf.nn.conv2d( X, self.discrim_W1, strides=[1,2,2,1], padding='SAME' ))
        h1 = tf.concat([h1, yb * tf.ones([self.batch_size, 12, 12, self.dim_y])],3)

        h2 = lrelu(batchnormalize( tf.nn.conv2d( h1, self.discrim_W2, strides=[1,2,2,1], padding='SAME')) )
        h2 = tf.reshape(h2, [self.batch_size, -1])
        h2 = tf.concat([h2, Y], 1)
        discri=tf.matmul(h2, self.discrim_W3 )
        h3 = lrelu(batchnormalize(discri))
        return h3


    def generate(self, Z, Y):

        yb = tf.reshape(Y, [self.batch_size, 1, 1, self.dim_y])
        Z = tf.concat([Z,Y],1)
        h1 = tf.nn.relu(batchnormalize(tf.matmul(Z, self.gen_W1)))
        h1 = tf.concat([h1, Y],1)
        h2 = tf.nn.relu(batchnormalize(tf.matmul(h1, self.gen_W2)))
        h2 = tf.reshape(h2, [self.batch_size,6,6,self.dim_W2])
        h2 = tf.concat([h2, yb*tf.ones([self.batch_size, 6,6, self.dim_y])],3)

        output_shape_l3 = [self.batch_size,12,12,self.dim_W3]
        h3 = tf.nn.conv2d_transpose(h2, self.gen_W3, output_shape=output_shape_l3, strides=[1,2,2,1])
        h3 = tf.nn.relu( batchnormalize(h3) )
        h3 = tf.concat([h3, yb*tf.ones([self.batch_size, 12, 12, self.dim_y])], 3)

        output_shape_l4 = [self.batch_size,24,24,self.dim_channel]
        h4 = tf.nn.conv2d_transpose(h3, self.gen_W4, output_shape=output_shape_l4, strides=[1,2,2,1])
        return h4


    def samples_generator(self, batch_size):
        Z = tf.placeholder(tf.float32, [batch_size, self.dim_z])
        Y = tf.placeholder(tf.float32, [batch_size, self.dim_y])

        yb = tf.reshape(Y, [batch_size, 1, 1, self.dim_y])
        Z_ = tf.concat([Z,Y], 1)
        h1 = tf.nn.relu(batchnormalize(tf.matmul(Z_, self.gen_W1)))
        h1 = tf.concat([h1, Y], 1)
        h2 = tf.nn.relu(batchnormalize(tf.matmul(h1, self.gen_W2)))
        h2 = tf.reshape(h2, [batch_size,6, 6,self.dim_W2])
        h2 = tf.concat([h2, yb*tf.ones([batch_size, 6,6, self.dim_y])], 3)

        output_shape_l3 = [batch_size,12, 12,self.dim_W3]
        h3 = tf.nn.conv2d_transpose(h2, self.gen_W3, output_shape=output_shape_l3, strides=[1,2,2,1])
        h3 = tf.nn.relu( batchnormalize(h3) )
        h3 = tf.concat([h3, yb*tf.ones([batch_size, 12,12,self.dim_y])], 3)

        output_shape_l4 = [batch_size,24, 24,self.dim_channel]
        h4 = tf.nn.conv2d_transpose(h3, self.gen_W4, output_shape=output_shape_l4, strides=[1,2,2,1])
        x = tf.nn.sigmoid(h4)
        return Z,Y,x
def OneHot(X, n, negative_class=0.):
    X = np.asarray(X).flatten()
    if n is None:
        n = np.max(X) + 1
    Xoh = np.ones((len(X), n)) * negative_class
    Xoh[np.arange(len(X)), X] = 1.
    return Xoh



'''def crop_resize(image_path, resize_shape=(64,64)):
    image = cv2.imread(image_path)
    height, width, channel = image.shape

    if width == height:
        resized_image = cv2.resize(image, resize_shape)
    elif width > height:
        resized_image = cv2.resize(image, (int(width * float(resize_shape[0])//height), resize_shape[1]))
        cropping_length = int( (resized_image.shape[1] - resize_shape[0]) // 2)
        resized_image = resized_image[:,cropping_length:cropping_length+resize_shape[1]]
    else:
        resized_image = cv2.resize(image, (resize_shape[0], int(height * float(resize_shape[1])/width)))
        cropping_length = int( (resized_image.shape[0] - resize_shape[1]) // 2)
        resized_image = resized_image[cropping_length:cropping_length+resize_shape[0], :]

    return resized_image/127.5 - 1
    '''

def save_visualization(X, nh_nw, save_path='./vis/sample.jpg'):
    h,w = X.shape[1], X.shape[2]
    img = np.zeros((h * nh_nw[0], w * nh_nw[1], 3))

    for n,x in enumerate(X):
        j = n // nh_nw[1]
        i = n % nh_nw[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = x

    scipy.misc.imsave(save_path, img)


n_epochs = 70
learning_rate = 0.0002
batch_size = 32
image_shape = [24, 24, 1]
dim_z = 100
dim_W1 = 1024
dim_W2 = 128
dim_W3 = 64
dim_channel = 1
k = 3
import csv

trX, trY, teX, teY, forecastX = load_wind_data_new()
print("shape of training samples ", shape(trX))
print("Wind data loaded")


def construct(X):
    X_new1 = np.copy(X[:, 288:576])
    X_new_high = [x * 1.2 for x in X_new1]
    X_new_low = [x * 0.8 for x in X_new1]
    x_samples_high = np.concatenate((X[:, 0:288], X_new_high), axis=1)
    x_samples_high = np.clip(x_samples_high, 0.05, 0.95)
    x_samples_low = np.concatenate((X[:, 0:288], X_new_low), axis=1)
    x_samples_low = np.clip(x_samples_low, 0.05, 0.9)
    return x_samples_high, x_samples_low


def construct2(X):
    X_new = X[:, 288:576]
    X_new_high = [x * 2.5 for x in X_new]
    # X_new_high=np.ones([32,288])
    X_new_low = [x * 0.4 for x in X_new]
    # X_new_low=np.zeros([32,288])
    X_new_high = np.clip(X_new_high, 0.16, 1)
    x_samples_high = np.concatenate((X[:, 0:288], X_new_high), axis=1)
    X_new_low = np.clip(X_new_low, 0, 0.6)
    x_samples_low = np.concatenate((X[:, 0:288], X_new_low), axis=1)
    return x_samples_high, x_samples_low


def construct_hard(X):
    x_samples_high = np.ones(shape(X), dtype=float)
    x_samples_low = np.zeros(shape(X), dtype=float)
    for i in range(len(X)):
        m = np.mean(X[i, 0:288])
        x_samples_high[i, :] = 4 * m * x_samples_high[i, :]
        x_samples_low[i, :] = 0.2 * m * x_samples_high[i, :]
    x_samples_high = np.clip(x_samples_high, 0, 1)
    return x_samples_high, x_samples_low


def plot(samples, X_real):
    m = 0
    f, axarr = plt.subplots(4, 8)
    for i in range(4):
        for j in range(8):
            axarr[i, j].plot(samples[m], linewidth=3.0)
            axarr[i, j].plot(X_real[m], 'r')
            axarr[i, j].set_xlim([0, 576])
            axarr[i, j].set_ylim([0, 16])
            m += 1
    plt.title('Comparison of predicted(blue) and real (red)')
    plt.savefig('comparison.png', bbox_inches='tight')
    plt.show()
    return f


def plot_sample(samples):
    m = 0
    f, axarr = plt.subplots(4, 8)
    for i in range(4):
        for j in range(8):
            axarr[i, j].plot(samples[m])
            axarr[i, j].set_xlim([0, 576])
            axarr[i, j].set_ylim([0, 16])
            m += 1
    plt.title('Generated samples')
    plt.savefig('generated_samples.png', bbox_inches='tight')
    plt.show()
    return f


dcgan_model = DCGAN(
    batch_size=batch_size,
    image_shape=image_shape,
    dim_z=dim_z,
    # W1,W2,W3: the dimension for convolutional layers
    dim_W1=dim_W1,
    dim_W2=dim_W2,
    dim_W3=dim_W3,
)
print("DCGAN model loaded")

Z_tf, Y_tf, image_tf, d_cost_tf, g_cost_tf, p_real, p_gen, \
complete_loss, high_tf, low_tf, mask_tf, log_loss, loss_former, loss_prepare = dcgan_model.build_model()

discrim_vars = filter(lambda x: x.name.startswith('discrim'),
                      tf.trainable_variables())
gen_vars = filter(lambda x: x.name.startswith('gen'), tf.trainable_variables())
discrim_vars = [i for i in discrim_vars]
gen_vars = [i for i in gen_vars]

train_op_discrim = (
    tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(-d_cost_tf,
                                                           var_list=discrim_vars))
train_op_gen = (
    tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(g_cost_tf,
                                                           var_list=gen_vars))
Z_tf_sample, Y_tf_sample, image_tf_sample = dcgan_model.samples_generator(
    batch_size=batch_size)

Z_np_sample = np.random.uniform(-1, 1, size=(batch_size, dim_z))
Y_np_sample = OneHot(np.random.randint(5, size=[batch_size]), n=5)
iterations = 0
P_real = []
P_fake = []
P_distri = []
discrim_loss = []

with tf.Session() as sess:
    # begin training
    init = tf.global_variables_initializer()
    sess.run(init)
    saver = tf.train.Saver()
    '''ckpt = tf.train.get_checkpoint_state('model.ckpt')
    print("CKPt", ckpt)
    if ckpt and ckpt.model_checkpoint_path:
      saver.restore(sess, 'model.ckpt')
      print(" [*] Success to read!")
    else: print("model load failed: here")'''
    # saver.restore(sess, 'model.ckpt.data-00000-of-00001')

    print("Number of batches in each epoch:", len(trY) / batch_size)
    for epoch in range(n_epochs):
        print("epoch" + str(epoch))
        index = np.arange(len(trY))
        np.random.shuffle(index)
        trX = trX[index]
        trY = trY[index]
        trY2 = OneHot(trY, n=5)
        for start, end in zip(
                range(0, len(trY), batch_size),
                range(batch_size, len(trY), batch_size)
        ):

            Xs = trX[start:end].reshape([-1, 24, 24, 1])
            Ys = trY2[start:end]
            # use uniform distribution data to generate adversarial samples
            Zs = np.random.uniform(-1, 1, size=[batch_size, dim_z]).astype(
                np.float32)

            # for each iteration, generate g and d respectively, k=2
            if np.mod(iterations, k) != 0:
                _, gen_loss_val = sess.run(
                    [train_op_gen, g_cost_tf],
                    feed_dict={
                        Z_tf: Zs,
                        Y_tf: Ys
                    })


            else:
                _, discrim_loss_val = sess.run(
                    [train_op_discrim, d_cost_tf],
                    feed_dict={
                        Z_tf: Zs,
                        Y_tf: Ys,
                        image_tf: Xs
                    })
                # gen_loss_val, p_real_val, p_gen_val = sess.run([g_cost_tf, p_real, p_gen], feed_dict={Z_tf:Zs, image_tf:Xs, Y_tf:Ys})
            p_real_val, p_gen_val = sess.run([p_real, p_gen],
                                             feed_dict={Z_tf: Zs, image_tf: Xs,
                                                        Y_tf: Ys})
            P_real.append(p_real_val.mean())
            P_fake.append(p_gen_val.mean())
            # discrim_loss.append(discrim_loss_val)

            if np.mod(iterations, 5000) == 0:
                print("iterations ", iterations)
                gen_loss_val, discrim_loss_val, p_real_val, p_gen_val = sess.run(
                    [g_cost_tf, d_cost_tf, p_real, p_gen],
                    feed_dict={Z_tf: Zs, image_tf: Xs,
                               Y_tf: Ys})
                print("Average P(real)=", p_real_val.mean())
                print("Average P(gen)=", p_gen_val.mean())
                print("discrim loss:", discrim_loss_val)
                print("gen loss:", gen_loss_val)

                Z_np_sample = np.random.uniform(-1, 1,
                                                size=(batch_size, dim_z))
                generated_samples = sess.run(
                    image_tf_sample,
                    feed_dict={
                        Z_tf_sample: Z_np_sample,
                        Y_tf_sample: Y_np_sample
                    })
                generated_samples = generated_samples.reshape([-1, 576])
                generated_samples = generated_samples * 16
                # save_visualization(generated_samples, (8, 8), save_path='./test/sample_' + str(iterations) + '.jpg')
                with open('%s.csv' % iterations, 'w')as csvfile:
                    # csvfile=file('%s.csv'%iterations, 'wb')
                    writer = csv.writer(csvfile)
                    writer.writerows(generated_samples)
            iterations = iterations + 1
    '''plt.plot(P_real)
    plt.plot(P_fake)
    plt.show()'''

    save_path = saver.save(sess,
                           'model.ckpt'
                           )
    print("Model saved in path: %s" % save_path)

    print("Start to generate scenarios")
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    lr = 0.001
    iterations = 0

    completed_samples = []
    mask = np.ones([batch_size, 24, 24, 1])
    mask[:, 12:24, :, :] = 0.0

    for start, end in zip(
            range(0, len(forecastX), batch_size),
            range(batch_size, len(forecastX), batch_size)
    ):
        print("ready to generate scenarios in iteration %s", iterations)
        forecast_samples = forecastX[start:end]
        Xs = teX[start:end]
        X_feed_high, X_feed_low = construct(forecast_samples)
        X_feed_high2, X_feed_low2 = construct2(forecast_samples)
        Ys = teY[start:end]
        Ys = OneHot(Ys, n=5)

        with open('orig_iter%s.csv' % iterations, 'w') as csvfile:
            # csvfile = file('orig_iter%s.csv' % iterations, 'wb')
            writer = csv.writer(csvfile)
            orig_samples = Xs * 16
            writer.writerows(orig_samples)
        with open('forecast_iter%s.csv' % iterations, 'w') as csvfile:
            # csvfile = file('forecast_iter%s.csv' % iterations, 'wb')
            writer = csv.writer(csvfile)
            orig_samples = forecast_samples * 16
            writer.writerows(orig_samples)
        with open('forhigh_iter%s.csv' % iterations, 'w') as csvfile:
            # csvfile = file('forhigh_iter%s.csv' % iterations, 'wb')
            writer = csv.writer(csvfile)
            orig_samples = X_feed_high2 * 16
            writer.writerows(orig_samples)
        with open('forlow_iter%s.csv' % iterations, 'w') as csvfile:
            # csvfile = file('forlow_iter%s.csv' % iterations, 'wb')
            writer = csv.writer(csvfile)
            orig_samples = X_feed_low2 * 16
            writer.writerows(orig_samples)

        # '''first plot
        # plt.plot(X_feed_high[0],'b')
        # plt.plot(X_feed_low[0],'r')
        # plt.plot(Xs[0],'g')
        # plt.show()#'''

        '''fig = plt.figure()
        fig.set_figheight(40)
        fig.set_figwidth(80)
        for m in range(32):
            ax = fig.add_subplot(4, 8, m + 1)
            ax.plot(orig_samples[m], color='b')
            ax.plot(X_feed_high2[m]*16, color='g')
            ax.plot(X_feed_low2[m]*16, color='y')'''

        Xs_shaped = Xs.reshape([-1, 24, 24, 1])
        samples = []

        for batch in range(50):  # number of batches
            print("Batch:", batch)
            zhats = np.random.uniform(-1, 1, size=[batch_size, dim_z]).astype(
                np.float32)
            image_pre = np.zeros([batch_size, 576])
            for i in range(batch_size):
                for j in range(288, 576):
                    image_pre[i][j] = np.random.uniform(X_feed_low[i, j],
                                                        X_feed_high[i, j])

            image_pre = image_pre.reshape([-1, 24, 24, 1])
            m = 0
            v = 0
            for i in range(1200):
                fd = {
                    Z_tf: zhats,
                    image_tf: image_pre,
                    Y_tf: Ys,
                    mask_tf: mask,
                }

                g, = sess.run([loss_prepare], feed_dict=fd)

                m_prev = np.copy(m)
                v_prev = np.copy(v)
                m = beta1 * m_prev + (1 - beta1) * g[0]
                v = beta2 * v_prev + (1 - beta2) * np.multiply(g[0], g[0])
                m_hat = m / (1 - beta1 ** (i + 1))
                v_hat = v / (1 - beta2 ** (i + 1))

                zhats += - np.true_divide(lr * m_hat, (np.sqrt(v_hat) + eps))
                zhats = np.clip(zhats, -1, 1)

                '''if np.mod(i, 500) == 0:
                    print("Gradient iteration:", i)'''

            image_pre = image_pre.reshape([-1, 576])

            '''plt.plot(generated_samples[0])
            plt.plot(image_pre[0]*16)
            plt.show()'''

            m = 0
            v = 0

            for i in range(1000):
                fd = {
                    Z_tf: zhats,
                    image_tf: Xs_shaped,
                    Y_tf: Ys,
                    high_tf: X_feed_high2.reshape([-1, 24, 24, 1]),
                    low_tf: X_feed_low2.reshape([-1, 24, 24, 1]),
                    mask_tf: mask,
                }

                g, log_loss_value, sample_loss_value = sess.run(
                    [complete_loss, log_loss, loss_former], feed_dict=fd)

                m_prev = np.copy(m)
                v_prev = np.copy(v)
                m = beta1 * m_prev + (1 - beta1) * g[0]
                v = beta2 * v_prev + (1 - beta2) * np.multiply(g[0], g[0])
                m_hat = m / (1 - beta1 ** (i + 1))
                v_hat = v / (1 - beta2 ** (i + 1))

                zhats += - np.true_divide(lr * m_hat, (np.sqrt(v_hat) + eps))
                zhats = np.clip(zhats, -1, 1)

                # if np.mod(i, 200) == 0:
                # print("Gradient iteration:", i)
                # print("Log loss", log_loss_value[0])
                # print("Sample loss", sample_loss_value)

                '''generated_samples = sess.run(
                    image_tf_sample,
                    feed_dict={
                        Z_tf_sample: zhats,
                        Y_tf_sample: Ys
                    })

                generated_samples = generated_samples.reshape(32, 576)
                generated_samples = generated_samples * 16
                plt.plot(generated_samples[0],'r')
                plt.plot(image_pre[0]*16, 'k')
                #plt.plot(generated_samples[1],'r')
                plt.plot(X_feed_high2[0]*16,'y')
                plt.plot(X_feed_low2[0]*16,'y')
                plt.plot(orig_samples[0],'b')
                #plt.plot(orig_samples[1],'b')
                plt.plot(X_feed_low[0]*16,'g')
                #plt.plot(X_feed_low[1] * 16, 'g')
                plt.plot(X_feed_high[0] * 16, 'g')
                #plt.plot(X_feed_high[1] * 16, 'g')
                plt.show()'''

            generated_samples = sess.run(
                image_tf_sample,
                feed_dict={
                    Z_tf_sample: zhats,
                    Y_tf_sample: Ys
                })

            generated_samples = generated_samples.reshape(32, 576)
            samples.append(generated_samples)
            # the following 5 lines were orginially commented out
            # plt.plot(generated_samples[0],color='r')
            # plt.plot(X_feed_low[0]*16, color='g')
            # plt.plot(X_feed_high[0]*16, color='y')
            # plt.plot(orig_samples[0], color='b')
            # plt.show()
            '''csvfile = file('generated_iter%sgroup%s.csv' % (iterations, batch), 'wb')
            writer = csv.writer(csvfile) 
            writer.writerows(generated_samples)'''

            '''for m in range(32):
                ax2 = fig.add_subplot(4, 8, m + 1)
                ax2.plot(generated_samples[m], color='r')



        fig.savefig('generated_iter%s.png'% (iterations))
        plt.close(fig)
        iterations += 1'''
        samples = np.array(samples, dtype=float)
        '''print(shape(samples))
        samples=samples.reshape([-1,12])
        samples=np.mean(samples,axis=1)
        samples=samples.reshape([-1,48])'''
        print(shape(samples))
        samples = samples * 16
        with open('generated_iter%s.csv' % iterations, 'w') as csvfile:
            # csvfile = file('generated_iter%s.csv' % iterations, 'wb')
            writer = csv.writer(csvfile)
            writer.writerows(samples.reshape([-1, 576]))
        # saver.save(sess, 'gans_model')
        iterations += 1
