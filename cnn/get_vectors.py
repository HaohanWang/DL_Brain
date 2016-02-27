__author__ = 'haohanwang'

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

from logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer

import pickle
import numpy as np
import numpy
from utility import loadData

import time

# def gm(v, num):
#     l = []
#     for i in range(num):
#         l.append(v)
#     return np.mat(l)
#
# params = pickle.load(open('../model/cnn.pkl'))
#
# m = []
# for param in params:
#     m.append(param.get_value(True))
# m.reverse()
# for i in range(0, len(m)-1, 2):
#     tmp = m[i]
#     m[i] = m[i+1]
#     m[i+1] = tmp
#
# train, dev, test = loadData.load_data()
# x = np.mat(train[0])
# r, c = x.shape
#
# for i in range(0, len(m)-1, 2):
#     x = x*np.mat(m[i])
#     x = x-gm(m[i+1], r)
#     x = np.tanh(x)
#
# print x.shape

class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):
        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        # pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                   numpy.prod(poolsize))
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape
        )

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height

        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]


def classify_lenet5(batch_size=500, output_size=20):
    """ Demonstrates lenet on MNIST dataset

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: path to the dataset used for training /testing (MNIST here)

    :type nkerns: list of ints
    :param nkerns: number of kernels on each layer
    """

    rng = numpy.random.RandomState(23455)


    # start-snippet-1
    x = T.matrix('x')  # the data is presented as rasterized images
    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # Reshape matrix of rasterized images of shape (batch_size, 28 * 28)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    # (28, 28) is the size of MNIST images.
    layer0_input = x.reshape((batch_size, 1, 37, 23))

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (28-5+1 , 28-5+1) = (24, 24)
    # maxpooling reduces this further to (24/2, 24/2) = (12, 12)
    # 4D output tensor is thus of shape (batch_size, nkerns[0], 12, 12)
    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 1, 37, 23),
        filter_shape=(20, 1, 4, 2),
        poolsize=(2, 2),
    )

    # layer1 = LeNetConvPoolLayer(
    #     rng,
    #     input=layer0.output,
    #     image_shape=(batch_size, 20, 17, 11),
    #     filter_shape=(50, 20, 4, 2),
    #     poolsize=(2, 2),
    # )
    #
    # layer4 = LeNetConvPoolLayer(
    #     rng,
    #     input=layer1.output,
    #     image_shape=(batch_size, 50, 7, 5),
    #     filter_shape=(100, 50, 4, 2),
    #     poolsize=(2, 2),
    # )

    layer2_input = layer0.output.flatten(2)

    # construct a fully-connected sigmoidal layer
    layer2 = HiddenLayer(
        rng,
        input=layer2_input,
        n_in=3740,
        n_out=output_size,
        activation=T.tanh,
        use_bias=True
    )

    # layer5 = HiddenLayer(
    #     rng,
    #     input=layer2.output,
    #     n_in=200,
    #     n_out=output_size,
    #     activation=T.tanh,
    #     use_bias=True
    # )

    # classify the values of the fully-connected sigmoidal layer
    layer3 = LogisticRegression(input=layer2.output, n_in=output_size, n_out=2)

    model_params = pickle.load(open('../model/cnn_dist_'+str(output_size)+'.pkl'))
    #
    layer0.W = theano.shared(
        value=numpy.array(
            model_params[2].get_value(True),
            dtype=theano.config.floatX
        ),
        name='W',
        borrow=True
    )

    layer0.b = theano.shared(
        value=numpy.array(
            model_params[3].get_value(True),
            dtype=theano.config.floatX
        ),
        name='b',
        borrow=True
    )

    # layer1.W = theano.shared(
    #     value=numpy.array(
    #         model_params[-4].get_value(True),
    #         dtype=theano.config.floatX
    #     ),
    #     name='W',
    #     borrow=True
    # )
    #
    # layer1.b = theano.shared(
    #     value=numpy.array(
    #         model_params[-3].get_value(True),
    #         dtype=theano.config.floatX
    #     ),
    #     name='b',
    #     borrow=True
    # )
    #
    # layer4.W = theano.shared(
    #     value=numpy.array(
    #         model_params[-6].get_value(True),
    #         dtype=theano.config.floatX
    #     ),
    #     name='W',
    #     borrow=True
    # )
    #
    # layer4.b = theano.shared(
    #     value=numpy.array(
    #         model_params[-5].get_value(True),
    #         dtype=theano.config.floatX
    #     ),
    #     name='b',
    #     borrow=True
    # )

    layer2.W = theano.shared(
        value=numpy.array(
            model_params[0].get_value(True),
            dtype=theano.config.floatX
        ),
        name='W',
        borrow=True
    )

    layer2.b = theano.shared(
        value=numpy.array(
            model_params[1].get_value(True),
            dtype=theano.config.floatX
        ),
        name='b',
        borrow=True
    )

    # layer5.W = theano.shared(
    #     value=numpy.array(
    #         model_params[-10].get_value(True),
    #         dtype=theano.config.floatX
    #     ),
    #     name='W',
    #     borrow=True
    # )
    #
    # layer5.b = theano.shared(
    #     value=numpy.array(
    #         model_params[-9].get_value(True),
    #         dtype=theano.config.floatX
    #     ),
    #     name='b',
    #     borrow=True
    # )

    layer3.W = theano.shared(
        value=numpy.array(
            model_params[4].get_value(True),
            dtype=theano.config.floatX
        ),
        name='W',
        borrow=True
    )

    layer3.b = theano.shared(
        value=numpy.array(
            model_params[5].get_value(True),
            dtype=theano.config.floatX
        ),
        name='b',
        borrow=True
    )

    # params = layer3.params + layer5.params + layer2.params +  layer4.params + layer1.params + layer0.params

    datasets = load_data(None)

    sets = ['train', 'dev', 'test']
    dimension = [20000, 20000, 20000]
    for k in range(3):
        if k == 0:
            classify_set_x, classify_set_y, classify_set_z, classify_set_m, classify_set_c, classify_set_b= datasets[k]
        else:
            classify_set_x, classify_set_y, classify_set_z= datasets[k]

        # compute number of minibatches for training, validation and testing
        n_classify_batches = classify_set_x.get_value(borrow=True).shape[0]
        n_classify_batches /= batch_size

        # allocate symbolic variables for the data
        index = T.lscalar()  # index to a [mini]batch
        classify = theano.function(
                [index],
                layer2.output,
                givens={
                    x: classify_set_x[index * batch_size: (index + 1) * batch_size],
                }
            )

        r = []

        for i in xrange(n_classify_batches):
            m = classify(i)
            r.extend(m)
        r = np.array(r)
        print r.shape
        r = np.append(r, np.reshape(classify_set_y.eval(),(dimension[k], 1)), 1)
        numpy.savetxt('../extractedInformation/cnn_dist_'+str(output_size)+'/'+sets[k]+'.csv', r, delimiter=",")

for k in [20, 40, 60, 80, 100]:
    start = time.time()
    classify_lenet5(output_size=k)
    print time.time() - start