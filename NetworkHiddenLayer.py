
from theano import tensor as T
from NetworkBaseLayer import Layer
from ActivationFunctions import strtoact


class HiddenLayer(Layer):
  recurrent = False

  def __init__(self, activation="tanh", **kwargs):
    """
    :type activation: str | list[str]
    """
    kwargs.setdefault("layer_class", "hidden")
    super(HiddenLayer, self).__init__(**kwargs)
    self.set_attr('activation', activation.encode("utf8"))
    self.activation = strtoact(activation)
    self.W_in = [self.add_param(self.create_forward_weights(s.attrs['n_out'],
                                                            self.attrs['n_out'],
                                                            name="W_in_%s_%s" % (s.name, self.name)))
                 for s in self.sources]
    self.set_attr('from', ",".join([s.name for s in self.sources]))


class ForwardLayer(HiddenLayer):
  def __init__(self, sparse_window = 1, **kwargs):
    kwargs.setdefault("layer_class", "hidden")
    super(ForwardLayer, self).__init__(**kwargs)
    self.set_attr('sparse_window', sparse_window) # TODO this is ugly
    self.attrs['n_out'] = sparse_window * kwargs['n_out']
    self.z = 0
    assert len(self.sources) == len(self.masks) == len(self.W_in)
    for s, m, W_in in zip(self.sources, self.masks, self.W_in):
      if s.attrs['sparse']:
        self.z += W_in[T.cast(s.output, 'int32')].reshape((s.output.shape[0],s.output.shape[1],s.output.shape[2] * W_in.shape[1]))
      elif m is None:
        self.z += self.dot(s.output, W_in)
      else:
        self.z += self.dot(self.mass * m * s.output, W_in)
    if not any(s.attrs['sparse'] for s in self.sources):
      self.z += self.b
    self.make_output(self.z if self.activation is None else self.activation(self.z))


class StateToAct(ForwardLayer):
  def __init__(self, dual=False, **kwargs):
    kwargs['n_out'] = 1
    kwargs.setdefault("layer_class", "state_to_act")
    super(StateToAct, self).__init__(**kwargs)
    self.set_attr("dual", dual)
    self.params = {}
    #self.make_output(T.concatenate([s.act[-1][-1] for s in self.sources], axis=-1).dimshuffle('x',0,1).repeat(self.sources[0].output.shape[0], axis=0))
    self.act = [ T.concatenate([s.act[i][-1] for s in self.sources], axis=-1).dimshuffle('x',0,1) for i in xrange(len(self.sources[0].act)) ]
    self.attrs['n_out'] = sum([s.attrs['n_out'] for s in self.sources])
    if dual:
      self.make_output(self.act[1])
      self.act[0] = T.tanh(self.act[1])
    else:
      self.make_output(self.act[0])
    self.index = T.ones((1, self.index.shape[1]), dtype = 'int8')


class DualStateLayer(ForwardLayer):
  def __init__(self, acts = "relu", acth = "tanh", **kwargs):
    kwargs.setdefault("layer_class", "dual")
    super(DualStateLayer, self).__init__(**kwargs)
    self.set_attr('acts', acts)
    self.set_attr('acth', acth)
    self.activations = [strtoact(acth), strtoact(acts)]
    self.params = {}
    self.W_in = []
    self.act = [self.b,self.b]
    for s,m in zip(self.sources,self.masks):
      assert len(s.act) == 2
      for i,a in enumerate(s.act):
        self.W_in.append(self.add_param(self.create_forward_weights(s.attrs['n_out'],
                                                                    self.attrs['n_out'],
                                                                    name="W_in_%s_%s_%d" % (s.name, self.name, i))))
        if s.attrs['sparse']:
          self.act[i] += self.W_in[-1][T.cast(s.act[i], 'int32')].reshape((s.act[i].shape[0],s.act[i].shape[1],s.act[i].shape[2] * self.W_in[-1].shape[1]))
        elif m is None:
          self.act[i] += self.dot(s.act[i], self.W_in[-1])
        else:
          self.act[i] += self.dot(self.mass * m * s.act[i], self.W_in[-1])
    for i in xrange(2):
      self.act[i] = self.activations[i](self.act[i])
    self.make_output(self.act[0])


class StateLayer(DualStateLayer):
  def __init__(self, acts = "relu", **kwargs):
    kwargs.setdefault("layer_class", "state")
    kwargs['acth'] = 'identity'
    super(StateToAct, self).__init__(acts, **kwargs)
    #self.make_output(T.concatenate([s.act[-1][-1] for s in self.sources], axis=-1).dimshuffle('x',0,1).repeat(self.sources[0].output.shape[0], axis=0))
    self.act[0] = T.tanh(self.act[1])
    self.make_output(self.act[0])
    self.index = T.ones((1, self.index.shape[1]), dtype = 'int8')
    self.attrs['n_out'] = sum([s.attrs['n_out'] for s in self.sources])


class BaseInterpolationLayer(ForwardLayer): # takes a base defined over T and input defined over T' and outputs a T' vector built over an input dependent linear combination of the base elements
  def __init__(self, base=None, method="softmax", **kwargs):
    assert base, "missing base in " + kwargs['name']
    kwargs['n_out'] = 1
    kwargs.setdefault("layer_class", "base")
    super(BaseInterpolationLayer, self).__init__(**kwargs)
    self.set_attr('base', ",".join([b.name for b in base]))
    self.set_attr('method', method)
    self.W_base = [ self.add_param(self.create_forward_weights(bs.attrs['n_out'], 1, name='W_base_%s_%s' % (bs.attrs['n_out'], self.name)), name='W_base_%s_%s' % (bs.attrs['n_out'], self.name)) for bs in base ]
    self.base = T.concatenate([b.output for b in base], axis=2) # TBD
    # self.z : T'
    bz = 0 # : T
    for x,W in zip(base, self.W_base):
      bz += T.dot(x.output,W) # TB1
    z = bz.reshape((bz.shape[0],bz.shape[1])).dimshuffle('x',1,0) + self.z.reshape((self.z.shape[0],self.z.shape[1])).dimshuffle(0,1,'x') # T'BT
    h = z.reshape((z.shape[0] * z.shape[1], z.shape[2])) # (T'xB)T
    if method == 'softmax':
      h_e = T.exp(h).dimshuffle(1,0)
      w = (h_e / T.sum(h_e, axis=0)).dimshuffle(1,0).reshape(z.shape).dimshuffle(2,1,0,'x').repeat(self.base.shape[2], axis=3) # TBT'D
      #w = T.nnet.softmax(h).reshape(z.shape).dimshuffle(2,1,0,'x').repeat(self.base.shape[2], axis=3) # TBT'D
    else:
      assert False, "invalid method %s in %s" % (method, self.name)
    
    self.set_attr('n_out', sum([b.attrs['n_out'] for b in base]))
    self.make_output(T.sum(self.base.dimshuffle(0,1,'x',2).repeat(z.shape[0], axis=2) * w, axis=0, keepdims=False).dimshuffle(1,0,2)) # T'BD


class ChunkingLayer(ForwardLayer): # Time axis reduction like in pLSTM described in http://arxiv.org/pdf/1508.01211v1.pdf
  def __init__(self, chunk_size=1, **kwargs):
    assert chunk_size >= 1
    kwargs['n_out'] = sum([s.attrs['n_out'] for s in kwargs['sources']]) * chunk_size
    kwargs.setdefault("layer_class", "chunking")
    super(ChunkingLayer, self).__init__(**kwargs)
    self.set_attr('chunk_size', chunk_size)
    z = T.concatenate([s.output for s in self.sources], axis=2) # BTD
    calloc = T.alloc(numpy.cast[theano.config.floatX](0), self.index.shape[0] + chunk_size - (self.index.shape[0] % chunk_size), z.shape[1], z.shape[2])
    container = T.set_subtensor(
      calloc[:self.index.shape[0]],
      z).dimshuffle(1,0,2) # BT'D
    ialloc = T.alloc(numpy.cast['int32'](1), self.index.shape[0] + chunk_size - (self.index.shape[0] % chunk_size), self.index.shape[1])
    self.index = T.set_subtensor(
      ialloc[:self.index.shape[0]],
      self.index)[::chunk_size] # BT'D

    #self.index = self.index.repeat(self.index.shape[0] % chunk_size, axis = 0)
    self.make_output(container.reshape((container.shape[0], container.shape[1]/chunk_size, container.shape[2] * chunk_size)).dimshuffle(1,0,2)) # T'BD


import theano
from theano.tensor.nnet import conv
import numpy

class ConvPoolLayer(ForwardLayer):
  def __init__(self, dx, dy, fx, fy, **kwargs):
    kwargs.setdefault("layer_class", "convpool")
    kwargs['n_out'] = fx * fy
    super(ConvPoolLayer, self).__init__(**kwargs)
    self.set_attr('dx', dx) # receptive fields
    self.set_attr('dy', dy)
    self.set_attr('fx', fx) # receptive fields
    self.set_attr('fy', fy)

    # instantiate 4D tensor for input
    n_in = numpy.sum([s.output for s in self.sources])
    assert n_in == dx * dy
    x_in  = T.concatenate([s.output for s in self.sources], axis = -1).dimshuffle(0,1,2,'x').reshape(self.sources[0].shape[0], self.sources[0].shape[1],dx, dy)
    range = 1.0 / numpy.sqrt(dx*dy)
    self.W = self.add_param(theano.shared( numpy.asarray(self.rng.uniform(low=-range,high=range,size=(2,1,fx,fy)), dtype = theano.config.floatX), name = "W_%s" % self.name), name = "W_%s" % self.name)
    conv_out = conv.conv2d(input, W)

    # initialize shared variable for weights.
    w_shp = (2, 3, 9, 9)
    w_bound = numpy.sqrt(3 * 9 * 9)
    W = theano.shared( numpy.asarray(
                rng.uniform(
                    low=-1.0 / w_bound,
                    high=1.0 / w_bound,
                    size=w_shp),
                dtype=input.dtype), name ='W')

    # initialize shared variable for bias (1D tensor) with random values
    # IMPORTANT: biases are usually initialized to zero. However in this
    # particular application, we simply apply the convolutional layer to
    # an image without learning the parameters. We therefore initialize
    # them to random values to "simulate" learning.
    b_shp = (2,)
    b = theano.shared(numpy.asarray(
                rng.uniform(low=-.5, high=.5, size=b_shp),
                dtype=input.dtype), name ='b')

    # build symbolic expression that computes the convolution of input with filters in w
    conv_out = conv.conv2d(input, W)

    # build symbolic expression to add bias and apply activation function, i.e. produce neural net layer output
    # A few words on ``dimshuffle`` :
    #   ``dimshuffle`` is a powerful tool in reshaping a tensor;
    #   what it allows you to do is to shuffle dimension around
    #   but also to insert new ones along which the tensor will be
    #   broadcastable;
    #   dimshuffle('x', 2, 'x', 0, 1)
    #   This will work on 3d tensors with no broadcastable
    #   dimensions. The first dimension will be broadcastable,
    #   then we will have the third dimension of the input tensor as
    #   the second of the resulting tensor, etc. If the tensor has
    #   shape (20, 30, 40), the resulting tensor will have dimensions
    #   (1, 40, 1, 20, 30). (AxBxC tensor is mapped to 1xCx1xAxB tensor)
    #   More examples:
    #    dimshuffle('x') -> make a 0d (scalar) into a 1d vector
    #    dimshuffle(0, 1) -> identity
    #    dimshuffle(1, 0) -> inverts the first and second dimensions
    #    dimshuffle('x', 0) -> make a row out of a 1d vector (N to 1xN)
    #    dimshuffle(0, 'x') -> make a column out of a 1d vector (N to Nx1)
    #    dimshuffle(2, 0, 1) -> AxBxC to CxAxB
    #    dimshuffle(0, 'x', 1) -> AxB to Ax1xB
    #    dimshuffle(1, 'x', 0) -> AxB to Bx1xA
    output = T.nnet.sigmoid(conv_out + b.dimshuffle('x', 0, 'x', 'x'))

    # create theano function to compute filtered images
    f = theano.function([input], output)
