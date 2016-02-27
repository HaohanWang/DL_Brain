__author__ = 'haohanwang'

"""This tutorial introduces the LeNet5 neural network architecture
using Theano.  LeNet5 is a convolutional neural network, good for
classifying images. This tutorial shows how to build the architecture,
and comes with all the hyper-parameters you need to reproduce the
paper's MNIST results.


This implementation simplifies the model in the following ways:

 - LeNetConvPool doesn't implement location-specific gain and bias parameters
 - LeNetConvPool doesn't implement pooling by average, it implements pooling
   by max.
 - Digit classification is implemented with a logistic regression rather than
   an RBF network
 - LeNet5 was not fully-connected convolutions at second layer

References:
 - Y. LeCun, L. Bottou, Y. Bengio and P. Haffner:
   Gradient-Based Learning Applied to Document
   Recognition, Proceedings of the IEEE, 86(11):2278-2324, November 1998.
   http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf

"""
import os
import sys
import time

import numpy
import numpy as np

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

from logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer, DropoutHiddenLayer, _dropout_from_layer

import pickle


def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2), convDropRate=0., poolDropRate=0.):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """

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
        if convDropRate > 0:
            conv_out = _dropout_from_layer(rng, conv_out, p=convDropRate)

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )

        if poolDropRate > 0:
            pooled_out = _dropout_from_layer(rng, pooled_out, p=poolDropRate)

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height

        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]


def evaluate_lenet5(learning_rate=0.05, n_epochs=20,
                    dataset='mnist.pkl.gz',
                    nkerns=None, batch_size=500, output_size=20):
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

    datasets = load_data(dataset)

    train_set_x, train_set_y, train_set_z, train_set_m, train_set_c, train_set_b = datasets[0]
    valid_set_x, valid_set_y, train_set_z = datasets[1]
    test_set_x, test_set_y, train_set_z = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size
    n_valid_batches /= batch_size
    n_test_batches /= batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # start-snippet-1
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
    z = T.ivector('z')
    m_1 = T.ivector('m_1')
    m_2 = T.ivector('m_2')
    m_3 = T.ivector('m_3')
    m_4 = T.ivector('m_4')
    m_5 = T.ivector('m_5')

    c_1 = T.ivector('c_1')
    c_2 = T.ivector('c_2')
    c_3 = T.ivector('c_3')
    c_4 = T.ivector('c_4')
    c_5 = T.ivector('c_5')

    b_1 = T.ivector('b_1')
    b_2 = T.ivector('b_2')
    b_3 = T.ivector('b_3')
    b_4 = T.ivector('b_4')
    b_5 = T.ivector('b_5')

    # [int] labels

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
        convDropRate=-1
    )

    # params_firstLayer = pickle.load(open('../model/firstLayer.pkl'))
    # layer0.W = theano.shared(
    #     value=numpy.array(
    #         params_firstLayer[2].get_value(True),
    #         dtype=theano.config.floatX
    #     ),
    #     name='W',
    #     borrow=True
    # )
    #
    # layer0.b = theano.shared(
    #     value=numpy.array(
    #         params_firstLayer[3].get_value(True),
    #         dtype=theano.config.floatX
    #     ),
    #     name='b',
    #     borrow=True
    # )

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (12-5+1, 12-5+1) = (8, 8)
    # maxpooling reduces this further to (8/2, 8/2) = (4, 4)
    # 4D output tensor is thus of shape (batch_size, nkerns[1], 4, 4)
    # layer1 = LeNetConvPoolLayer(
    #     rng,
    #     input=layer0.output,
    #     image_shape=(batch_size, 20, 17, 11),
    #     filter_shape=(50, 20, 4, 2),
    #     poolsize=(2, 2),
    #     convDropRate=-1
    # )
    #
    # layer4 = LeNetConvPoolLayer(
    #     rng,
    #     input=layer1.output,
    #     image_shape=(batch_size, 50, 7, 5),
    #     filter_shape=(100, 50, 4, 2),
    #     poolsize=(2, 2),
    #     convDropRate=-1
    # )

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (batch_size, nkerns[1] * 4 * 4),
    # or (500, 50 * 4 * 4) = (500, 800) with the default values.
    layer2_input = layer0.output.flatten(2)

    # construct a fully-connected sigmoidal layer
    layer2 = DropoutHiddenLayer(
        rng,
        input=layer2_input,
        n_in=3740,
        n_out=output_size,
        activation=T.tanh,
        dropout_rate=0.5,
        use_bias=True
    )

    # layer5 = DropoutHiddenLayer(
    #     rng,
    #     input=layer2.output,
    #     n_in=1000,
    #     n_out=output_size,
    #     activation=T.tanh,
    #     dropout_rate=0.5,
    #     use_bias=True
    # )

    layer_MF = DropoutHiddenLayer(
        rng,
        input=layer2.output,
        n_in=output_size,
        n_out=10,
        activation=T.tanh,
        dropout_rate=0.5,
        use_bias=True
    )

    layer_CC = DropoutHiddenLayer(
        rng,
        input=layer2.output,
        n_in=output_size,
        n_out=10,
        activation=T.tanh,
        dropout_rate=0.5,
        use_bias=True
    )

    layer_BP = DropoutHiddenLayer(
        rng,
        input=layer2.output,
        n_in=output_size,
        n_out=10,
        activation=T.tanh,
        dropout_rate=0.5,
        use_bias=True
    )



    # classify the values of the fully-connected sigmoidal layer
    layer3 = LogisticRegression(input=layer2.output, n_in=output_size, n_out=2)
    layer6 = LogisticRegression(input=layer2.output, n_in=output_size, n_out=2)

    layer_MF_1 = LogisticRegression(input=layer_MF.output, n_in=10, n_out=10)
    layer_MF_2 = LogisticRegression(input=layer_MF.output, n_in=10, n_out=10)
    layer_MF_3 = LogisticRegression(input=layer_MF.output, n_in=10, n_out=10)
    layer_MF_4 = LogisticRegression(input=layer_MF.output, n_in=10, n_out=10)
    layer_MF_5 = LogisticRegression(input=layer_MF.output, n_in=10, n_out=10)

    layer_CC_1 = LogisticRegression(input=layer_CC.output, n_in=10, n_out=10)
    layer_CC_2 = LogisticRegression(input=layer_CC.output, n_in=10, n_out=10)
    layer_CC_3 = LogisticRegression(input=layer_CC.output, n_in=10, n_out=10)
    layer_CC_4 = LogisticRegression(input=layer_CC.output, n_in=10, n_out=10)
    layer_CC_5 = LogisticRegression(input=layer_CC.output, n_in=10, n_out=10)

    layer_BP_1 = LogisticRegression(input=layer_BP.output, n_in=10, n_out=10)
    layer_BP_2 = LogisticRegression(input=layer_BP.output, n_in=10, n_out=10)
    layer_BP_3 = LogisticRegression(input=layer_BP.output, n_in=10, n_out=10)
    layer_BP_4 = LogisticRegression(input=layer_BP.output, n_in=10, n_out=10)
    layer_BP_5 = LogisticRegression(input=layer_BP.output, n_in=10, n_out=10)

    params = layer2.params +  layer0.params + layer3.params + layer6.params + \
        layer_MF.params + layer_MF_1.params + layer_MF_2.params + layer_MF_3.params + layer_MF_4.params + layer_MF_5.params + \
        layer_CC.params + layer_CC_1.params + layer_CC_2.params + layer_CC_3.params + layer_CC_4.params + layer_CC_5.params + \
        layer_BP.params + layer_BP_1.params + layer_BP_2.params + layer_BP_3.params + layer_BP_4.params + layer_BP_5.params
    # print params[1].get_value(True)
    # for i in range(len(params)):
    #     params[i] = model_params[i]
    # print params[1].get_value(True)

    best_params = params

    learning_rates = [0.5*0.1, 0.5*0.1, 1*0.1, 1*0.1] + [0.5*0.1] *40
    reg_rate = [0.5, 0.5, 0.5, 0.5] + [0.1]*40

    # learning_rates = [0.005, 0.005, 0.01, 0.01, 0.01, 0.01, 0.15, 0.15, 0.15, 0.15, 0.25, 0.25]*100
    # reg_rate = [1, 1, 1, 1, 1.5, 1.5, 1.5, 1.5, 3, 3, 3, 3]
    # current best: lr = 0.05, L1_reg = 5, L2_reg = 5
    # learning_rates = [1, 1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
    # reg_rate = [1, 1, 5, 5, 5, 5, 5, 5, 5, 5]
    # for 20 states
    # learning_rates = [0.5*0.01, 0.5*0.01, 1*0.01, 1*0.01, 1*0.01, 1*0.01, 1.5*0.01, 1.5*0.01, 1.5*0.01, 1.5*0.01, 2.5*0.01, 2.5*0.01]
    # reg_rate = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3]
    # for 60 states
    # learning_rates = [0.1*0.02, 0.1*0.02, 0.1*0.02, 0.1*0.02, 0.1*0.02, 0.1*0.02, 0.1*0.02, 0.1*0.02, 1.5*0.02, 1.5*0.02, 1.5*0.02, 1.5*0.02]
    # reg_rate = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.5, 0.5]
    # for 80 states
    # learning_rates = [0.1*0.02, 0.1*0.02, 0.1*0.02, 0.1*0.02, 0.1*0.02, 0.1*0.02, 0.1*0.02, 0.1*0.02, 1.5*0.02, 1.5*0.02, 1.5*0.02, 1.5*0.02]
    # reg_rate = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.5, 0.5]
    # for 100 states
    # learning_rates = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.5, 1.5, 1.5, 1.5]
    # reg_rate = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.5, 0.5]

    L1 = (
        sum([reg_rate[i]*(params[i]).sum() for i in range(len(params))])
    )

    L2 = (
        sum([reg_rate[i]*(params[i]**2).sum() for i in range(len(params))])
    )

    # the cost we minimize during training is the NLL of the model
    # cost = layer3.negative_log_likelihood(y) + L1 + L2

    mf_cost = layer_MF_1.negative_log_likelihood(m_1) + layer_MF_2.negative_log_likelihood(m_2) + \
        layer_MF_3.negative_log_likelihood(m_3) + layer_MF_4.negative_log_likelihood(m_4) + layer_MF_5.negative_log_likelihood(m_5)

    cc_cost = layer_CC_1.negative_log_likelihood(c_1) + layer_CC_2.negative_log_likelihood(c_2) + \
        layer_CC_3.negative_log_likelihood(c_3) + layer_CC_4.negative_log_likelihood(c_4) + layer_CC_5.negative_log_likelihood(c_5)

    bp_cost = layer_BP_1.negative_log_likelihood(b_1) + layer_BP_2.negative_log_likelihood(b_2) + \
        layer_BP_3.negative_log_likelihood(b_3) + layer_BP_4.negative_log_likelihood(b_4) + layer_BP_5.negative_log_likelihood(b_5)

    cost = mf_cost + cc_cost + bp_cost + \
        layer2.distance_form_2(y) + layer3.negative_log_likelihood(y) + layer6.negative_log_likelihood(z) + L1/100.0

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # create a list of all model parameters to be fit by gradient descent
    # params = layer3.params + layer2.params + layer0.params


    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    updates = [
        (param_i, param_i - lr * grad_i)
        for param_i, grad_i, lr in zip(params, grads, learning_rates)
    ]

    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size],
            z: train_set_z[index * batch_size: (index + 1) * batch_size],
            m_1: train_set_m[index * batch_size: (index + 1) * batch_size, 0],
            m_2: train_set_m[index * batch_size: (index + 1) * batch_size, 1],
            m_3: train_set_m[index * batch_size: (index + 1) * batch_size, 2],
            m_4: train_set_m[index * batch_size: (index + 1) * batch_size, 3],
            m_5: train_set_m[index * batch_size: (index + 1) * batch_size, 4],
            b_1: train_set_b[index * batch_size: (index + 1) * batch_size, 0],
            b_2: train_set_b[index * batch_size: (index + 1) * batch_size, 1],
            b_3: train_set_b[index * batch_size: (index + 1) * batch_size, 2],
            b_4: train_set_b[index * batch_size: (index + 1) * batch_size, 3],
            b_5: train_set_b[index * batch_size: (index + 1) * batch_size, 4],
            c_1: train_set_c[index * batch_size: (index + 1) * batch_size, 0],
            c_2: train_set_c[index * batch_size: (index + 1) * batch_size, 1],
            c_3: train_set_c[index * batch_size: (index + 1) * batch_size, 2],
            c_4: train_set_c[index * batch_size: (index + 1) * batch_size, 3],
            c_5: train_set_c[index * batch_size: (index + 1) * batch_size, 4],
        }
    )
    # end-snippet-1

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'
    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
    # found
    improvement_threshold = 0.995  # a relative improvement of this much is
    # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
    # go through this many
    # minibatche before checking the network
    # on the validation set; in this case we
    # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        print params[1].get_value(True)
        for minibatch_index in xrange(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % 100 == 0:
                print 'training @ iter = ', iter
            cost_ij = train_model(minibatch_index)

            if (iter + 1) % validation_frequency == 0:
                print 'last minibatch cost', cost_ij
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    # improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss * \
                            improvement_threshold:
                        patience = max(patience, iter * patience_increase)
                        best_params = params

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [
                        test_model(i)
                        for i in xrange(n_test_batches)
                    ]
                    test_score = numpy.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

    save_object(best_params, '../model/cnn_dist_'+str(output_size)+'.pkl')


if __name__ == '__main__':
    for oz in [20, 40, 60, 80, 100]:
        evaluate_lenet5(output_size=oz)
    # params = pickle.load(open('../model/cnn.pkl'))
    # for param in params:
    #     print param.get_value(True).shape