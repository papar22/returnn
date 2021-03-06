import tensorflow as tf
from TFNetworkLayer import _ConcatInputLayer, get_concat_sources_data_template, Loss
from TFUtil import Data
import numpy as np
import time
import os


class HMMFactorization(_ConcatInputLayer):

  layer_class = "hmm_factorization"

  def __init__(self, attention_weights, base_encoder_transformed, prev_state, prev_outputs, n_out, debug=False,
               attention_location=None, threshold=None, transpose_and_average_att_weights=False, top_k=None,
               window_size=None, first_order_alignments=False, first_order_k=None, window_factor=1.0,
               tie_embedding_weights=None, prev_prev_state=None, **kwargs):
    """
    HMM factorization as described in Parnia Bahar's paper.
    Out of rnn loop usage.
    Please refer to the demos to see the layer in use.
    :param LayerBase attention_weights: Attention weights of shape [I, J, B, 1]
    :param LayerBase base_encoder_transformed: Encoder, inner most dimension transformed to a constant size
    'intermediate_size'. Tensor of shape [J, B, intermediate_size]
    :param LayerBase prev_state: Previous state data, with the innermost dimension set to a constant size
    'intermediate_size'. Tensor of shape [I, B, intermediate_size].
    :param LayerBase prev_outputs: Previous output data with the innermost dimension set to a constant size
    'intermediate_size'. Tensor of shape [I, B, intermediate_size]
    :param bool debug: True/False, whether to print debug info or not
    :param float|None threshold: (float, >0), if not set to 'none', all attention values below this threshold will be
    set to 0. Slightly improves speed.
    :param bool transpose_and_average_att_weights: Set to True if using Transformer architecture. So, if
    attention_weights are of shape [J, B, H, I] with H being the amount of heads in the architecture. We will then
    average out over the heads to get the final attention values used.
    :param int n_out: Size of output dim (usually not set manually)
    TODO top_k documentation
    :param kwargs:
    """

    super(HMMFactorization, self).__init__(**kwargs)

    self.iteration = 0
    self.batch_iteration = 0

    in_loop = True if len(prev_state.output.shape) == 1 else False

    # Get data
    if in_loop is False:
      self.attention_weights = attention_weights.output.get_placeholder_as_time_major()  # [J, B, H/1, I]
      self.base_encoder_transformed = base_encoder_transformed.output.get_placeholder_as_time_major()  # [J, B, f]
      self.prev_state = prev_state.output.get_placeholder_as_time_major()  # [I, B, f]
      self.prev_outputs = prev_outputs.output.get_placeholder_as_time_major()  # [I, B, f]
      self.prev_prev_state = prev_prev_state.output.get_placeholder_as_time_major()  # [I, B, f]
    else:
      self.attention_weights = attention_weights.output.get_placeholder_as_batch_major()  # [B, J, 1]
      self.base_encoder_transformed = base_encoder_transformed.output.get_placeholder_as_batch_major()  # [B, J, f]
      self.prev_state = prev_state.output.get_placeholder_as_batch_major()  # [B, intermediate_size]
      self.prev_outputs = prev_outputs.output.get_placeholder_as_batch_major()  # [B, intermediate_size]
      self.prev_prev_state = prev_prev_state.output.get_placeholder_as_batch_major()  # [B, f]

    if debug:
      self.attention_weights = tf.Print(self.attention_weights, [tf.shape(self.attention_weights)],
                                        message='Attention weight shape: ', summarize=100)

    if debug:
      self.base_encoder_transformed = tf.Print(self.base_encoder_transformed, [tf.shape(self.base_encoder_transformed),
                                                                               tf.shape(self.prev_state),
                                                                               tf.shape(self.prev_outputs)],
                                               message='Shapes of base encoder, prev_state and prev_outputs pre shaping: ',
                                               summarize=100)

    # Transpose and average out attention weights (for when we use transformer architecture)
    if transpose_and_average_att_weights is True:

      if in_loop is False:
        # attention_weights is [J, B, H, I]
        self.attention_weights = tf.transpose(self.attention_weights, perm=[3, 0, 1, 2])  # Now it is [I, J, B, H]
        self.attention_weights = tf.reduce_mean(self.attention_weights, keep_dims=True, axis=3)  # Now [I, J, B, 1]
      else:
        # attention_weights is [J, B, H, 1?]
        self.attention_weights = tf.squeeze(self.attention_weights, axis=3)
        self.attention_weights = tf.reduce_mean(self.attention_weights, keep_dims=True, axis=2)
        self.attention_weights = tf.transpose(self.attention_weights, perm=[1, 0, 2])  # Now it is [J, B, 1]

      if debug:
        self.attention_weights = tf.Print(self.attention_weights, [tf.shape(self.attention_weights)],
                                          message='Attention weight shape after transposing and avg: ', summarize=100)
    else:
      if in_loop is True:
        # Edge case of attention weights
        self.attention_weights = tf.transpose(self.attention_weights, perm=[1, 0, 2])  # Now [J, B, 1]
        if debug:
          self.attention_weights = tf.Print(self.attention_weights, [tf.shape(self.attention_weights)],
                                            message='Attention weight shape after transposing: ', summarize=100)

    # top_k management
    assert isinstance(top_k, int) or isinstance(top_k, str), "HMM factorization: top_k of wrong format"

    if first_order_k is None and top_k is not None:
      first_order_k = top_k

    # if we want dynamic top_k
    if isinstance(top_k, str):
      import TFUtil
      vs = vars(TFUtil).copy()
      vs.update({"tf": tf, "self": self})
      top_k = eval(top_k, vs)
      if debug:
        top_k = tf.Print(top_k, [top_k], message="Dynamically calculated top k (may be overriden): ")

    # max cut top_k
    if in_loop:
      top_k = tf.minimum(top_k, tf.shape(self.base_encoder_transformed)[1])
    else:
      top_k = tf.minimum(top_k, tf.shape(self.base_encoder_transformed)[0])

    # if we want dynamic top_k
    if isinstance(first_order_k, str):
      import TFUtil
      vs = vars(TFUtil).copy()
      vs.update({"tf": tf, "self": self})
      first_order_k = eval(first_order_k, vs)
      if debug:
        first_order_k = tf.Print(first_order_k, [first_order_k], message="Dynamically calculated top k (may be overriden): ")

    # max cut top_k
    if in_loop:
      first_order_k = tf.minimum(first_order_k, tf.shape(self.base_encoder_transformed)[1])
    else:
      first_order_k = tf.minimum(first_order_k, tf.shape(self.base_encoder_transformed)[0])

    if first_order_alignments is True:

      # Get shape data
      if in_loop is False:
        s = tf.shape(self.base_encoder_transformed)
        time_i = tf.shape(self.prev_state)[0]
        batch_size = s[1]
        time_j = s[0]
        time_j_prime = time_j
        intermediate_size = s[-1]
        f = intermediate_size
      else:
        s = tf.shape(self.base_encoder_transformed)  # [B, J, f]
        batch_size = s[0]
        time_j = s[1]
        time_j_prime = time_j
        intermediate_size = s[-1]
        f = intermediate_size

      # In search, batch includes the beam for each actual sequence

      # Posterior attention
      prev_prev_output_and_decoder = self.prev_prev_state + self.prev_outputs  # [(I,) B, f]
      prev_prev_output_and_decoder_exp = tf.expand_dims(prev_prev_output_and_decoder, axis=-3)  # [(I,) 1, B, f]
      # Note: we use the auto-broadcast function of "+" to make the tiling ops

      encoder_tr = self.base_encoder_transformed  # [J, B, f] in_loop: [B, J, f]
      if in_loop is False:
        encoder_h1 = tf.expand_dims(encoder_tr, axis=0)  # [1, J', B, f]
      else:
        encoder_tr = tf.transpose(encoder_tr, perm=[1, 0, 2])  # [J, B, f]
        encoder_h1 = encoder_tr  # [J', B, f]
      encoder_h = encoder_h1  # [(1,) J, B, f]

      post_attention = prev_prev_output_and_decoder_exp + encoder_h1  # [(I,) J', B, f]
      post_attention = tf.layers.dense(post_attention, units=base_encoder_transformed.output.shape[-1],
                                       activation=tf.nn.tanh, use_bias=False)  # TODO: check if this is how we want it
      post_attention = tf.layers.dense(post_attention, units=1, activation=None, use_bias=False,
                                       name="post_att")  # [(I,) J', B, 1]
      post_attention = tf.nn.softmax(post_attention, axis=-3, name="post_att_softmax")  # [(I,) J', B, 1]

      # topk on posterior attention
      if in_loop is False:
        post_attention_topk = tf.transpose(post_attention, perm=[0, 2, 3, 1])  # [I, B, 1, J']
        post_attention_topk, post_top_indices = tf.nn.top_k(post_attention_topk, k=first_order_k)  # Both [I, B, 1, top_k]
        post_attention_topk = tf.squeeze(post_attention_topk, axis=-2)  # [I, B, top_k=J']
        post_top_indices = tf.squeeze(post_top_indices, axis=2)  # [I, B, top_k]
        ii, bb, _ = tf.meshgrid(tf.range(time_i), tf.range(batch_size), tf.range(first_order_k), indexing='ij')  # [I, B, k]
        post_indices = tf.stack([ii, bb, post_top_indices], axis=-1)  # [I, B, k, 3]

        if debug:
          post_indices = tf.Print(post_indices, [post_top_indices], message="post_top_indices", summarize=20)

        encoder_h2 = tf.tile(tf.expand_dims(tf.transpose(encoder_tr, perm=[1, 0, 2]), axis=0), [time_i, 1, 1, 1])  # [I, B, J, f]
        encoder_h2 = tf.gather_nd(encoder_h2, post_indices)  # [I, B, top_k=J', f]
        encoder_h2 = tf.expand_dims(encoder_h2, axis=1)  # [I, 1, B, top_k=J', f]
      else:
        post_attention_topk = tf.transpose(post_attention, perm=[1, 2, 0])  # [B, 1, J']
        post_attention_topk, post_top_indices = tf.nn.top_k(post_attention_topk, k=first_order_k)  # Both [B, 1, top_k]
        post_attention_topk = tf.squeeze(post_attention_topk, axis=-2)  # [B, top_k=J']
        post_top_indices = tf.squeeze(post_top_indices, axis=1)  # [B, top_k]
        bb, _ = tf.meshgrid(tf.range(batch_size), tf.range(first_order_k), indexing='ij')  # [B, k]
        post_indices = tf.stack([bb, post_top_indices], axis=-1)  # [B, k, 2]
        if debug:
          post_indices = tf.Print(post_indices, [post_indices, post_top_indices], message="post_indices, post_top_indices", summarize=20)
        encoder_h2 = tf.transpose(encoder_tr, perm=[1, 0, 2])  # [B, J, f]
        encoder_h2 = tf.gather_nd(encoder_h2, post_indices)  # [B, top_k=J', f]
        encoder_h2 = tf.expand_dims(encoder_h2, axis=0)  # [1, B, top_k=J', f]

      # First order attention
      prev_output_and_decoder = self.prev_state + self.prev_outputs  # [(I,) B, f]
      prev_output_and_decoder_exp = tf.expand_dims(prev_output_and_decoder, axis=-3)  # [(I,) 1, B, f]

      first_order_att = prev_output_and_decoder_exp + encoder_h  # Additive attention  [(I,) J, B, f]
      first_order_att = tf.expand_dims(first_order_att, axis=-2)  # [(I,) J, B, 1, f]
      first_order_att = first_order_att + encoder_h2  # [(I,) J, B, top_k=J', f]
      first_order_att = tf.layers.dense(first_order_att, units=base_encoder_transformed.output.shape[-1],
                                        activation=tf.nn.tanh, use_bias=False)  # TODO: check if this is how we want it
      first_order_att = tf.layers.dense(first_order_att, units=1, activation=None, use_bias=False)  # [(I,) J, B, top_k=J', 1]
      first_order_att = tf.nn.softmax(first_order_att, axis=-4, name="fo_softmax")  # [(I,) J, B, top_k=J', 1]
      first_order_att = tf.squeeze(first_order_att, axis=-1)  # [(I,) J, B, top_k=J']

      # Combine together
      if in_loop is False:
        self.attention_weights = tf.einsum('ibk,ijbk->ijbk', post_attention_topk,
                                           first_order_att)  # [I, J, B, top_k=J']
      else:
        self.attention_weights = tf.einsum('bk,jbk->jbk', post_attention_topk,
                                           first_order_att)  # [J, B, top_k=J']

      self.attention_weights = tf.reduce_sum(self.attention_weights, axis=-1, keep_dims=True)  # [(I,) J, B, 1]

      if debug:
        if in_loop:
          self.attention_weights = tf.Print(self.attention_weights,
                                            [tf.reduce_sum(tf.transpose(self.attention_weights, perm=[1, 0, 2]), axis=-2)[0],
                                             tf.transpose(self.attention_weights, perm=[1, 0, 2])[0]], summarize=1000,
                                            message="self.attention_weights sum and eg")
        else:
          self.attention_weights = tf.Print(self.attention_weights,
                                            [tf.reduce_sum(tf.transpose(self.attention_weights, perm=[0, 2, 1, 3]), axis=-2)[0],
                                             tf.transpose(self.attention_weights, perm=[0, 2, 1, 3])[0, 0]], summarize=1000,
                                            message="self.attention_weights sum and eg")
      if attention_location is not None:
        if in_loop:
          i = tf.get_default_graph().get_tensor_by_name('output/rec/while/Identity:0')  # super hacky, get current step
        else:
          i = None

        #self.attention_weights = tf.Print(self.attention_weights, [i, post_top_indices, tf.transpose(self.attention_weights, perm=[1, 0, 2])[0]], summarize=100, message="Attention:")
        if i is not None:
          self.attention_weights = tf.py_func(func=self.save_tensor, inp=[self.attention_weights, attention_location,
                                                                          self.network.global_train_step,
                                                                          post_attention, i],
                                              Tout=tf.float32, stateful=True)
        else:
          self.attention_weights = tf.py_func(func=self.save_tensor, inp=[self.attention_weights, attention_location,
                                                                          self.network.global_train_step,
                                                                          post_attention],
                                              Tout=tf.float32, stateful=True)

    # Use only top_k from self.attention_weights
    if top_k is not None:

      if in_loop is False:
        temp_attention_weights = tf.transpose(self.attention_weights, perm=[0, 2, 3, 1])  # Now [I, B, 1, J]
      else:
        temp_attention_weights = tf.transpose(self.attention_weights, perm=[1, 2, 0])  # Now [B, 1, J]
      # temp_attention_weights [(I,) B, 1, J]

      if window_size is not None:

        # window_size/2 is the size of window on each side. j=i is always seen

        assert window_size % 2 == 0, "HMM Factorization: Window size has to be divisible by 2!"

        if isinstance(top_k, int):
          assert top_k <= window_size, "HMM Factorization: top_k can be maximally as large as window_size!"

        if in_loop is False:
          # Example:
          # window_size = 2, I=J=4:
          # [0, 0] = 1, [0, 1] = 1, [0, 2] = 0, [0, 3] = 0
          # [1, 0] = 1, [1, 1] = 1, [1, 2] = 1, [1, 3] = 0
          # [2, 0] = 0, [2, 1] = 1, [2, 2] = 1, [2, 3] = 1
          # [3, 0] = 0, [3, 1] = 0, [3, 2] = 1, [3, 3] = 1

          # Get shapes
          sh = tf.shape(temp_attention_weights)
          ti = sh[0]
          tj = sh[3]
          b = sh[1]
          #mask = tf.matrix_band_part(tf.ones([ti, tj], dtype=tf.bool), int(window_size/2), int(window_size/2))  # [I, J]

          # Shifting with window_factor, quite hacky using a precomputed mask. If using longer sequences set max_i to
          # the length of your longest target sequence.
          def static_mask_graph(max_i=500):
            # make static mask, is only done once during graph creation
            max_j = np.int64(np.floor(max_i * window_factor))
            matrices = []
            for i in range(max_i):
              i = np.int64(np.floor(i * window_factor))
              new_m = np.concatenate([np.zeros([np.maximum(i - int(window_size/2), 0)], dtype=np.bool),
                            np.ones([window_size + 1 - np.maximum(-i + int(window_size/2), 0) - np.maximum((i + int(window_size/2)) - (max_j - 1), 0)], dtype=np.bool),
                            np.zeros([np.maximum(np.minimum(max_j - i - int(window_size/2), max_j) - 1, 0)], dtype=np.bool)],
                           axis=0)
              matrices.append(new_m)
            mask = np.stack(matrices)
            return mask

          mask = static_mask_graph()

          # The following lines are to make the behaviour of in loop and outer loop match
          f = tf.cast(tf.floor(tf.cast(tj, tf.float32) / window_factor), tf.int32)
          mask_1 = tf.slice(mask, [0, 0], [f, tj])
          mask_2 = tf.concat([tf.zeros([tj - int(window_size/2), ti - f], dtype=tf.bool),
                              tf.ones([int(window_size/2), ti - f], dtype=tf.bool)], axis=0)
          mask_2 = tf.transpose(mask_2, perm=[1, 0])
          mask = tf.concat([mask_1, mask_2], axis=0)

          # Then use expand_dims to make it 4D
          mask = tf.expand_dims(tf.expand_dims(mask, axis=1), axis=1)  # Now [I, 1, 1, J]
          mask = tf.tile(mask, [1, b, 1, 1])  # Now [I, B, 1, J]

          # Mask temp_attention_weights to 0
          temp_attention_weights = tf.where(mask, temp_attention_weights, tf.zeros_like(temp_attention_weights))
          # TODO: should we renormalize?
        else:

          sh = tf.shape(temp_attention_weights)
          tj = sh[2]
          b = sh[0]

          i = tf.get_default_graph().get_tensor_by_name('output/rec/while/Identity:0')  # super hacky, get current step
          i = tf.cast(tf.floor(tf.cast(i, tf.float32) * window_factor), tf.int32)  # Shift by window_factor
          i = tf.minimum(i, tj)

          mask = tf.concat([tf.zeros([tf.maximum(i - int(window_size/2), 0)], dtype=tf.bool),
                            tf.ones([window_size + 1 - tf.maximum(-i + int(window_size/2), 0) - tf.maximum((i + int(window_size/2)) - (tj - 1), 0)], dtype=tf.bool),
                            tf.zeros([tf.maximum(tf.minimum(tj - i - int(window_size/2), tj) - 1, 0)], dtype=tf.bool)],
                           axis=0)  # Shape [J]

          mask = tf.expand_dims(tf.expand_dims(mask, axis=0), axis=0)  # Now [1, 1, J]
          mask = tf.tile(mask, [b, 1, 1])

          # mask is now [B, 1, J], having true only in J where j=i +/- window_size
          # Mask temp_attention_weights to 0
          temp_attention_weights = tf.where(mask, temp_attention_weights, tf.zeros_like(temp_attention_weights))
          # TODO: should we renormalize?

      top_values, top_indices = tf.nn.top_k(temp_attention_weights, k=top_k) # top_values and indices [(I,) B, 1, top_k]

      if debug:
        top_indices = tf.Print(top_indices, [top_indices], message="top_indices eg", summarize=20)

      if in_loop is False:
        self.attention_weights = tf.transpose(top_values, perm=[0, 3, 1, 2])  # Now [I, J=top_k, B, 1]
      else:
        self.attention_weights = tf.transpose(top_values, perm=[2, 0, 1])  # Now [J=top_k, B, 1]

      if debug:
        self.attention_weights = tf.Print(self.attention_weights, [tf.shape(self.attention_weights)],
                                          message='Top K Attention weight shape: ', summarize=100)

    # Get data
    if in_loop is False:
      attention_weights_shape = tf.shape(self.attention_weights)
      time_i = attention_weights_shape[0]
      batch_size = attention_weights_shape[2]
      time_j = attention_weights_shape[1]
      intermediate_size = tf.shape(self.base_encoder_transformed)[-1]
    else:
      attention_weights_shape = tf.shape(self.attention_weights)
      batch_size = attention_weights_shape[1]
      time_j = attention_weights_shape[0]
      intermediate_size = tf.shape(self.base_encoder_transformed)[-1]

    # Convert base_encoder_transformed, prev_state and prev_outputs to correct shape
    if in_loop is False:
      self.base_encoder_transformed = tf.tile(tf.expand_dims(self.base_encoder_transformed, axis=0),
                                              [time_i, 1, 1, 1])  # [I, J, B, intermediate_size]

      self.prev_state = tf.tile(tf.expand_dims(self.prev_state, axis=1),
                                [1, time_j, 1, 1])  # [I, J, B, intermediate_size]

      self.prev_outputs = tf.tile(tf.expand_dims(self.prev_outputs, axis=1),
                                  [1, time_j, 1, 1])  # [I, J, B, intermediate_size]
    else:
      self.base_encoder_transformed = tf.transpose(self.base_encoder_transformed,
                                                   perm=[1, 0, 2])  # [J, B, f]

      self.prev_state = tf.tile(tf.expand_dims(self.prev_state, axis=0),
                                [time_j, 1, 1])  # [J, B, f]
      self.prev_outputs = tf.tile(tf.expand_dims(self.prev_outputs, axis=0),
                                  [time_j, 1, 1])  # [J, B, f]

    # Fix self.base_encoder_transformed if in top_k
    if top_k is not None:
      if in_loop is False:
        self.base_encoder_transformed = tf.transpose(self.base_encoder_transformed,
                                                     perm=[0, 2, 1, 3])  # Now [I, B, J, f]
        top_indices = tf.squeeze(top_indices, axis=2)
        ii, jj, _ = tf.meshgrid(tf.range(time_i), tf.range(batch_size), tf.range(top_k), indexing='ij')  # [I B k]

        # Stack complete index
        index = tf.stack([ii, jj, top_indices], axis=-1)  # [I B k 3]
        # index = tf.Print(index, [tf.shape(index)], message='index shape: ', summarize=100)
      else:
        self.base_encoder_transformed = tf.transpose(self.base_encoder_transformed,
                                                     perm=[1, 0, 2])  # Now [B, J, f]
        top_indices = tf.squeeze(top_indices, axis=1)
        jj, _ = tf.meshgrid(tf.range(batch_size), tf.range(top_k), indexing='ij')
        # Stack complete index
        index = tf.stack([jj, top_indices], axis=-1)

      # Get the same values again
      self.base_encoder_transformed = tf.gather_nd(self.base_encoder_transformed, index)

      if in_loop is False:
        self.base_encoder_transformed = tf.transpose(self.base_encoder_transformed,
                                                     perm=[0, 2, 1, 3])  # [I, J, B, f]
      else:
        self.base_encoder_transformed = tf.transpose(self.base_encoder_transformed,
                                                     perm=[1, 0, 2])  # [J, B, f]

    if debug:
      self.base_encoder_transformed = tf.Print(self.base_encoder_transformed, [tf.shape(self.base_encoder_transformed),
                                                                               tf.shape(self.prev_state),
                                                                               tf.shape(self.prev_outputs)],
                                               message='Shapes of base encoder, prev_state '
                                                       'and prev_outputs post shaping: ',
                                               summarize=100)

    # Permutate attention weights correctly
    if in_loop is False:
      self.attention_weights = tf.transpose(self.attention_weights, perm=[0, 2, 3, 1])  # Now [I, B, 1, J]
    else:
      # Before [J, B, 1]
      self.attention_weights = tf.transpose(self.attention_weights, perm=[1, 2, 0])  # Now [B, 1, J]

    if debug:
      self.attention_weights = tf.Print(self.attention_weights, [tf.shape(self.attention_weights)],
                                        message='attention_weights shape transposed: ',
                                        summarize=100)

      self.base_encoder_transformed = tf.Print(self.base_encoder_transformed,
                                               [tf.shape(self.base_encoder_transformed + self.prev_outputs + self.prev_state)],
                                               message='Pre lex logits shape: ', summarize=100)

    # Get logits, now [I, J, B, vocab_size]/[J, B, vocab_size]
    if tie_embedding_weights is None:
      lexicon_logits = tf.layers.dense(self.base_encoder_transformed + self.prev_outputs + self.prev_state,
                                       units=n_out,
                                       activation=None,
                                       use_bias=False)
    else:
      lexicon_weight = tf.get_default_graph().get_tensor_by_name(
                          tie_embedding_weights.get_base_absolute_name_scope_prefix() + "W:0")  # [vocab, emb]
      lexicon_weight = tf.transpose(lexicon_weight, perm=[1, 0])  # [emb, vocab]
      lexicon_logits = self.linear(x=self.base_encoder_transformed + self.prev_outputs + self.prev_state,
                                   weight=lexicon_weight,
                                   units=lexicon_weight.shape[1])  # [(I,) J, B, vocab_size]

    if debug:
      lexicon_logits = tf.Print(lexicon_logits, [tf.shape(lexicon_logits)], message='Post lex logits shape: ', summarize=100)

    if in_loop is False:
      lexicon_logits = tf.transpose(lexicon_logits, perm=[0, 2, 1, 3])  # Now [I, B, J, vocab_size]
    else:
      lexicon_logits = tf.transpose(lexicon_logits, perm=[1, 0, 2])  # Now [B, J, vocab_size]

    # TODO: add sampled softmax here

    # Now [I, B, J, vocab_size]/[B, J, vocab_size], Perform softmax on last layer
    lexicon_model = tf.nn.softmax(lexicon_logits)

    if debug:
      lexicon_model = tf.Print(lexicon_model, [tf.shape(lexicon_model)], message='lexicon_model shape: ', summarize=100)

    # in_loop=True   Multiply for final logits, [B, 1, J] x [B, J, vocab_size] ----> [B, 1, vocab]
    # in_loop=False: Multiply for final logits, [I, B, 1, J] x [I, B, J, vocab_size] ----> [I, B, 1, vocab]
    final_output = tf.matmul(self.attention_weights, lexicon_model)

    if debug:
      final_output = tf.Print(final_output, [tf.shape(final_output)], message='final_output shape: ', summarize=100)

    if in_loop is False:
      # Squeeze [I, B, vocab]
      final_output = tf.squeeze(final_output, axis=2)
    else:
      # Squeeze [B, vocab]
      final_output = tf.squeeze(final_output, axis=1)

    if debug:
      final_output = tf.Print(final_output, [tf.shape(final_output)], message='final_output post squeeze shape: ',
                              summarize=100)

    # TODO: check if it works with rnns
    self.output.placeholder = final_output
    # Warning: if any shaping errors occur, look into github history of how we previously got shaping info

    # Add all trainable params
    with self.var_creation_scope() as scope:
      self._add_all_trainable_params(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name))

  def linear(self, x, units, inp_dim=None, weight=None, bias=False):
    in_shape = tf.shape(x)
    inp = tf.reshape(x, [-1, in_shape[-1]])
    if weight is None:
      weight = tf.get_variable("lin_weight", trainable=True, shape=[inp_dim, units], dtype=tf.float32)
    out = tf.matmul(inp, weight)
    if bias:
      bias = tf.get_variable("lin_bias", trainable=True, initializer=tf.zeros_initializer, shape=[units],
                             dtype=tf.float32)
      out = out + bias
    out_shape = tf.concat([in_shape[:-1], [units]], axis=0)
    out = tf.reshape(out, out_shape)
    # TODO: maybe use tensordot instead
    return out

  def sampled_softmax(self, inp, weight, num_samples, full_vocab_size, full_softmax=False, sample_method="log"):
    # TODO: maybe add bias

    if full_softmax:
      # TODO: full softmax version
      logits = self.linear(x=inp, units=full_vocab_size, weight=weight)
      return tf.nn.softmax(logits, axis=-1)
    else:
      # TODO: flatten inp
      weight_shape = tf.shape(weight)  # [vocab_size, emb]
      input_shape = tf.shape(inp)
      inp = tf.reshape(inp, shape=[-1, weight_shape[1]])  # [B * I * k, emb]

      # TODO: maybe use version with true
      # TODO: get sample
      sampled = tf.random_uniform(shape=[num_samples], minval=0, maxval=tf.shape(inp)[0], dtype=tf.int32)  # [num_samples]

      # TODO: get weight for samples
      weight = tf.nn.embedding_lookup(weight, sampled)  # [num_samples, emb]

      # TODO: matmul
      # [B * I * k, emb] x [emb, num_samples] -> [B * I * k, num_samples]
      logits = tf.matmul(inp, weight, transpose_b=True)

      # TODO: reshape back
      output_shape = tf.concat([input_shape[:-1], [num_samples]], axis=0)
      logits = tf.reshape(logits, shape=output_shape)  # [I, B, k, num_samples] TODO: get exact shape

      # TODO: normalize with softmax
      distribution = tf.nn.softmax(logits, axis=-1)

      # TODO: project back onto full distribution

      sampled_shaped = tf.reshape(sampled, shape=input_shape[:-1])  # [I, B, k] TODO: get exact shape
      # TODO: get exact shape
      # TODO: debug this
      i, b, k, _ = tf.meshgrid(tf.range(input_shape[0]),
                               tf.range(input_shape[1]),
                               tf.range(input_shape[2]),
                               tf.range(num_samples),
                               indexing="ij")
      idx = tf.stack([i, b, k, sampled_shaped], axis=-1)
      full_out_shape = tf.concat([input_shape[:-1], [full_vocab_size]], axis=0)
      full_distribution = tf.scatter_nd(idx, distribution, shape=full_out_shape)
      return full_distribution

  def _add_all_trainable_params(self, tf_vars):
    for var in tf_vars:
      self.add_param(param=var, trainable=True, saveable=True)

  @classmethod
  def transform_config_dict(cls, d, network, get_layer):
    d["from"] = d["prev_state"]
    super(HMMFactorization, cls).transform_config_dict(d, network=network, get_layer=get_layer)
    d["attention_weights"] = get_layer(d["attention_weights"])
    d["base_encoder_transformed"] = get_layer(d["base_encoder_transformed"])
    d["prev_state"] = get_layer(d["prev_state"])
    d["prev_prev_state"] = get_layer(d["prev_prev_state"])
    d["prev_outputs"] = get_layer(d["prev_outputs"])
    if "tie_embedding_weights" in d:
      d["tie_embedding_weights"] = get_layer(d["tie_embedding_weights"])

  def save_tensor(self, attention_tensor, location, global_train_step, posterior_attention=None, i_step=None):
      # save tensor to file location
      d = {}
      d["i_step"] = i_step
      d["global_train_step"] = global_train_step
      d["shape"] = attention_tensor.shape
      d["attention_tensor"] = attention_tensor
      d["posterior_attention"] = posterior_attention

      if i_step is not None:
        if i_step == 0:
          self.batch_iteration += 1
      else:
        self.batch_iteration += 1

      np.save(str(location.decode("utf-8")) + '/' + str(self.batch_iteration) + "_" + str(i_step) +'_attention.npy', d)

      self.iteration += 1
      return attention_tensor


class SimpleHMMFactorization(_ConcatInputLayer):

  layer_class = "simple_hmm_factorization"

  def __init__(self, attention_weights, base_encoder_transformed, prev_state, prev_outputs, n_out, topk=5, **kwargs):

    super(SimpleHMMFactorization, self).__init__(**kwargs)


    self.attention_weights = attention_weights.output.get_placeholder_as_time_major()  # (I, J, bs, 1)
    self.base_encoder_transformed = base_encoder_transformed.output.get_placeholder_as_time_major() # (J, bs, f)
    self.prev_state = prev_state.output.get_placeholder_as_time_major() # (I, bs, f)
    self.prev_outputs = prev_outputs.output.get_placeholder_as_time_major() # (I, bs, f)

    attention_weights_shape = tf.shape(self.attention_weights)
    time_i = attention_weights_shape[0]
    batch_size = attention_weights_shape[2]
    seq_len = attention_weights_shape[1]

    # self.attention_weights = tf.Print(self.attention_weights, [tf.shape(self.attention_weights)],
    #                                       message='Attention weight shape: ', summarize=100)
    #
    #
    # self.base_encoder_transformed = tf.Print(self.base_encoder_transformed, [tf.shape(self.base_encoder_transformed)],
    #                                       message='base_encoder_transformed shape: ', summarize=100)
    #
    # self.prev_state = tf.Print(self.prev_state, [tf.shape(self.prev_state)],
    #                                       message='prev_state shape: ', summarize=100)
    #
    #
    # self.prev_outputs = tf.Print(self.prev_outputs, [tf.shape(self.prev_outputs)],
    #                                       message='prev_outputs shape: ', summarize=100)


    temp_k = tf.minimum(topk, seq_len)
    # temp_k = tf.Print(temp_k, [temp_k], message='temp_k: ', summarize=100)

    # Permutate attention weights correctly to work for top_k
    self.attention_weights = tf.transpose(self.attention_weights, perm=[0, 2, 3, 1])  # Now [I, bs, 1, J]

    # self.attention_weights = tf.Print(self.attention_weights, [tf.shape(self.attention_weights)],
    #                                        message='Attention weight shape: ', summarize=100)

    top_alignments_v, top_alignments_i  = tf.nn.top_k(self.attention_weights, k=temp_k) # Now [I, bs, 1, K]

    # top_alignments_i = tf.Print(top_alignments_i, [tf.shape(top_alignments_i)],
    #                                        message='top_alignments shape: ', summarize=100)




    encoder_shape=tf.shape(self.base_encoder_transformed) # (J, bs, f)


    ################### Extend the batch_tensor from (bsxk,1) --> (I, bsxk, 1) #################### Size are wrong
    #inter_bs_tensor = tf.tile(tf.expand_dims(tf.range(encoder_shape[1]),axis=1),[1,temp_k]) # (bs, K)
    # batch_tensor=tf.reshape(tf.tile(tf.expand_dims(inter_bs_tensor,axis=0),[time_i, 1, 1]),[time_i, encoder_shape[1]*temp_k,1]) # (I, bs*k, 1)
    # indices_tensor=tf.reshape(top_alignments_i,[time_i, batch_size*temp_k,1]) # (I, bs*k, 1)
    #
    # inter_bs_tensor = tf.tile(tf.expand_dims(tf.range(batch_size),axis=1),[1,temp_k]) # (bs, K)
    # batch_tensor=tf.reshape(tf.tile(tf.expand_dims(inter_bs_tensor,axis=0),[time_i, 1, 1]),[time_i*batch_size*temp_k,1]) # (I*bs*k, 1)


    # Make indices for the rest of axes
    ii, jj, kk, _ = tf.meshgrid( tf.range(time_i), tf.range(batch_size), tf.range(1), tf.range(temp_k), indexing='ij')
    # Stack complete index
    indices_tensor = tf.stack([ii, jj, kk, top_alignments_i], axis=-1)  # (I, B, 1, k)





    # batch_tensor=tf.reshape(tf.tile(tf.expand_dims(tf.range(batch_size),axis=1),[1,temp_k]),[batch_size*temp_k,1])
    # indices_tensor=tf.reshape(top_alignments_i,[-1,1])


    # batch_tensor = tf.Print(batch_tensor, [tf.shape(batch_tensor)],
    #                                       message='batch_tensor shape: ', summarize=100)


    # indices_tensor = tf.Print(indices_tensor, [tf.shape(indices_tensor)],
    #                                       message='indices_tensor shape: ', summarize=100)


    # Convert base_encoder_transformed, prev_state and prev_outputs to correct shape
    self.base_encoder_transformed = tf.tile(tf.expand_dims(self.base_encoder_transformed, axis=0),
                                              [time_i, 1, 1, 1])  # [I, J, B, intermediate_size]

    self.base_encoder_transformed = tf.transpose(self.base_encoder_transformed, perm=[0, 2, 3, 1])  # Now [I, B, f, J]


    #top_encoder_parts is the top 5 (or memory_time if memory_time < 5) most probable encoder states
    self.top_encoder_parts=tf.gather_nd(self.base_encoder_transformed, indices_tensor)

    # self.top_encoder_parts = tf.Print(self.top_encoder_parts, [tf.shape(self.top_encoder_parts)],
    #                                       message='+++++top_encoder_parts shape: ', summarize=100)



    # # Convert base_encoder_transformed, prev_state and prev_outputs to correct shape
    # self.base_encoder_transformed = tf.tile(tf.expand_dims(self.top_encoder_parts, axis=0),
    #                                           [time_i, 1, 1, 1])  # [I, J, B, intermediate_size]

    self.prev_state = tf.tile(tf.expand_dims(self.prev_state, axis=1),
                                [1, temp_k, 1, 1])  # [I, J, B, intermediate_size]

    self.prev_outputs = tf.tile(tf.expand_dims(self.prev_outputs, axis=1),
                                  [1, temp_k, 1, 1])  # [I, J, B, intermediate_size]

    #top_encoder_parts is the top 5 (or memory_time if memory_time < 5) most probable encoder states
    # self.top_encoder_parts=tf.gather_nd(self.base_encoder_transformed,tf.reshape(tf.concat([batch_tensor,indices_tensor],1),[time_i, temp_k,encoder_shape[1],-1])) #(I, K, bs, f)

    # Permutate attention weights correctly
    # self.attention_weights = tf.transpose(self.attention_weights, perm=[0, 2, 3, 1])  # Now  # (I, bs, 1, J)

    # self.attention_weights = tf.Print(self.attention_weights, [tf.shape(self.attention_weights)],
    #                                       message='+++++Attention weight shape: ', summarize=100)
    #
    #
    #
    #
    # self.base_encoder_transformed = tf.Print(self.base_encoder_transformed, [tf.shape(self.base_encoder_transformed)],
    #                                       message='++++base_encoder_transformed shape: ', summarize=100)
    #
    # self.prev_state = tf.Print(self.prev_state, [tf.shape(self.prev_state)],
    #                                       message='++++prev_state shape: ', summarize=100)
    #
    #
    # self.prev_outputs = tf.Print(self.prev_outputs, [tf.shape(self.prev_outputs)],
    #                                       message='++++prev_outputs shape: ', summarize=100)


    self.top_encoder_parts = tf.transpose(self.top_encoder_parts, perm=[0, 3, 1, 2])  # [I, B, f, K]



    # Get logits, now [I, K, B, vocab_size]/[J, B, vocab_size]   # [I, J, B, f]
    lexicon_logits = tf.layers.dense(self.top_encoder_parts + self.prev_outputs + self.prev_state,
                                     units=n_out,
                                     activation=None,
                                     use_bias=False)

    #
    # lexicon_logits = tf.Print(lexicon_logits, [tf.shape(lexicon_logits)],
    #                                       message='++++lexicon_logits shape: ', summarize=100)

    lexicon_logits = tf.transpose(lexicon_logits, perm=[0, 2, 1, 3])  # Now [I, B, J, |V|]

    # Now [I, B, J, vocab_size]/[B, J, vocab_size], Perform softmax on last layer
    lexicon_model = tf.nn.softmax(lexicon_logits)

    # in_loop=True   Multiply for final logits, [B, 1, J] x [B, J, vocab_size] ----> [B, 1, vocab]
    # in_loop=False: Multiply for final logits, [I, B, 1, J] x [I, B, J, vocab_size] ----> [I, B, 1, vocab]
    final_output = tf.matmul(top_alignments_v, lexicon_model)
    # Squeeze [I, B, vocab]
    final_output = tf.squeeze(final_output, axis=2)


    # Set shaping info
    output_size = self.input_data.size_placeholder[0]
    # final_output = tf.Print(final_output, [self.input_data.size_placeholder[0]],
    #                               message='Prev output size placeholder: ',
    #                               summarize=100)

    self.output.placeholder = final_output
    self.output.size_placeholder = {0: output_size}
    self.output.time_dim_axis = 0
    self.output.batch_dim_axis = 1


    # print('BEAAAAAAAAAAAAAAAAAAM SIZE:    ' + str(prev_state.output.beam_size))
    self.output.beam_size = prev_state.output.beam_size

    # Add all trainable params
    with self.var_creation_scope() as scope:
      self._add_all_trainable_params(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name))

  def _add_all_trainable_params(self, tf_vars):
    for var in tf_vars:
      self.add_param(param=var, trainable=True, saveable=True)

  @classmethod
  def transform_config_dict(cls, d, network, get_layer):
    d.setdefault("from", [d["attention_weights"]])
    super(SimpleHMMFactorization, cls).transform_config_dict(d, network=network, get_layer=get_layer)
    d["attention_weights"] = get_layer(d["attention_weights"])
    d["base_encoder_transformed"] = get_layer(d["base_encoder_transformed"])
    d["prev_state"] = get_layer(d["prev_state"])
    d["prev_outputs"] = get_layer(d["prev_outputs"])

  @classmethod
  def get_out_data_from_opts(cls, attention_weights, prev_state, n_out, out_type=None, sources=(), **kwargs):

    data = attention_weights.output
    data = data.copy_as_time_major()  # type: Data
    data.shape = (None, n_out)
    data.time_dim_axis = 0
    data.batch_dim_axis = 1
    data.dim = n_out

    return data


class GeometricNormalization(_ConcatInputLayer):

  layer_class = "geometric_normalization"

  def __init__(self, target_embedding_layer, n_out, **kwargs):
    super(GeometricNormalization, self).__init__(**kwargs)

    # TODO: add asserts

    decoder_output_dis = self.input_data.placeholder  # of shape [<?>,...,<?>, embedding_size]
    from TFUtil import nan_to_num
    decoder_output_dis = decoder_output_dis  # Remove nans in input_data?

    # TODO: make less hacky
    self.word_embeddings = tf.get_default_graph().get_tensor_by_name(
                        target_embedding_layer.get_base_absolute_name_scope_prefix() + "W:0")  # [vocab_size, embedding_size]

    # set shaping info correctly
    for d in range(len(decoder_output_dis.shape) - 1):  # -1 due to not wanting to add feature dim
      self.word_embeddings = tf.expand_dims(self.word_embeddings, axis=0)

    decoder_output_dis = tf.expand_dims(decoder_output_dis, axis=-2)  # [..., 1, embedding_size]

    distances = self.word_embeddings - decoder_output_dis  # [..., vocab_size, embedding_size]
    distances = tf.pow(distances, 2)  # [..., vocab_size, embedding_size]
    distances = tf.reduce_sum(distances, axis=-1)  # [..., vocab_size]
    max_distances = tf.reduce_max(distances, axis=-1, keepdims=True)  # [..., 1]
    distances = max_distances - distances  # [..., vocab_size]
    normalization_constant = 1 / tf.reduce_sum(distances, axis=-1, keepdims=True)  # [..., 1]
    output_geometric = tf.multiply(distances, normalization_constant)  # [..., vocab_size]

    if self.network.search_flag is False:
      from TFUtil import OutputWithActivation
      self.output_before_activation = OutputWithActivation(self.input_data.placeholder)  # Actually not doing activation here
    self.output.placeholder = output_geometric

  @classmethod
  def transform_config_dict(cls, d, network, get_layer):
    super(GeometricNormalization, cls).transform_config_dict(d, network=network, get_layer=get_layer)
    d["target_embedding_layer"] = get_layer(d["target_embedding_layer"])


class GeometricNormalizationLoss(Loss):
  """
  geometric_normalization loss function
  """
  class_name = "geometric_normalization_loss"

  def __init__(self, target_embedding_layer, min_regularizer=0.0, max_regularizer=0.0, debug=False, **kwargs):
    super(GeometricNormalizationLoss, self).__init__(**kwargs)
    # Get embedding weights
    self.embedding_weights = None
    self.target_embedding_layer = target_embedding_layer
    self.min_regularizer = min_regularizer
    self.max_regularizer = max_regularizer
    self.debug = debug

  def get_value(self):

    assert self.target.sparse, "GeometricNormalizationLoss: Supporting only sparse targets"

    # TODO: scopes
    # TODO: make less hacky
    self.embedding_weights = tf.get_default_graph().get_tensor_by_name(
                      self.target_embedding_layer.get_base_absolute_name_scope_prefix() + "W:0")  # [vocab_size, embedding_size]

    output_embs = self.output_with_activation.y

    if self.target_flat is not None:
      targets = self.target.placeholder
      output_embs = tf.transpose(output_embs, perm=[1, 0, 2])  # TODO: make this generic
    else:
      targets = self.target.copy_as_time_major().placeholder

    target_embeddings = tf.nn.embedding_lookup(self.embedding_weights, ids=targets)  # [B, I, embedding_size]
    out = tf.squared_difference(output_embs, target_embeddings)

    if self.target_flat is not None:
      #out = self.reduce_func(tf.reduce_mean(out, axis=1))
      out = self.reduce_func(out)
    else:
      out = self.reduce_func(out)

    # TODO: maybe use stop_gradients instead?
    if self.min_regularizer > 0.0:
      assert self.min_regularizer <= self.max_regularizer, "Geo soft: Check min/max reg setting!"

      # TODO: make batch_axis dynamic
      reg = self._regularizer(target_embds=target_embeddings, batch_axis=0)
      reg_factor = tf.reduce_sum(out)/tf.reduce_sum(reg)
      reg_factor = tf.Print(reg_factor, [reg_factor], "Raw reg_factor")
      reg_factor = tf.clip_by_value(reg_factor, clip_value_min=self.min_regularizer,
                                    clip_value_max=self.max_regularizer)
      if self.debug:
        out = tf.Print(out, [out, reg, reg_factor], message="Out, Regularizer, reg_factor: ")
      out = out - reg_factor * reg

    if self.debug:
      out = tf.Print(out, [tf.reduce_sum(target_embeddings)], message="Target emb sum: ")

    return out

  def _regularizer(self, target_embds, batch_axis=0):
    # Get average vector over batch
    target_avg = tf.reduce_mean(target_embds, axis=batch_axis)  # [1, I, embedding_size]
    dist = target_embds - target_avg  # [B, I, embedding_size]
    dist = tf.pow(dist, 2)  # [B, I, embedding_size]
    return tf.reduce_sum(dist)


  @classmethod
  def transform_config_dict(cls, d, network, get_layer):
    super(GeometricNormalizationLoss, cls).transform_config_dict(d, network=network, get_layer=get_layer)
    d["target_embedding_layer"] = get_layer(d["target_embedding_layer"])

  def get_error(self):
    # TODO: only get_error when actually needed
    return tf.constant(0.0)


class GeometricCrossEntropy(Loss):
  """
  geometric_normalization loss function
  """
  class_name = "geometric_ce_loss"

  def __init__(self, target_embedding_layer, full_vocab_size, vocab_sample_size=10, debug=False, **kwargs):
    super(GeometricCrossEntropy, self).__init__(**kwargs)
    # Get embedding weights
    self.embedding_weights = None
    self.target_embedding_layer = target_embedding_layer
    self.vocab_sample_size = vocab_sample_size
    self.debug = debug
    self.full_vocab_size = full_vocab_size

  def get_value(self):

    assert self.target.sparse, "GeometricNormalizationLoss: Supporting only sparse targets"

    # TODO: scopes
    # TODO: make less hacky
    self.embedding_weights = tf.get_default_graph().get_tensor_by_name(
                      self.target_embedding_layer.get_base_absolute_name_scope_prefix() + "W:0")  # [vocab_size, embedding_size]

    output_embs = self.output_with_activation.y

    if self.target_flat is not None:
      targets = self.target.placeholder
      output_embs = tf.transpose(output_embs, perm=[1, 0, 2])  # TODO: make this generic
    else:
      targets = self.target.copy_as_time_major().placeholder  # [B, I]

    output_embs = tf.expand_dims(output_embs, axis=-2)  # [B, I, 1, emb]

    # reshape targets to get to sampled vocab size
    targets = tf.expand_dims(targets, axis=-1)  # [B, I, 1]
    # sample shape
    sample_shape = tf.concat([tf.shape(targets)[:-1], [self.vocab_sample_size]], axis=0)

    # sample distribution to get random samples
    neg_sample = tf.random_uniform(shape=sample_shape, minval=0, maxval=self.full_vocab_size, dtype=tf.int32)
    # TODO: maybe force remove targets from sample

    # concat
    targets = tf.concat([targets, neg_sample], axis=-1)  # [B, I, vocab_sample_size + 1]

    target_embeddings = tf.nn.embedding_lookup(self.embedding_weights, ids=targets)  # [B, I, vocab_sample_size + 1, embedding_size]

    distances = tf.squared_difference(target_embeddings, output_embs)  # [..., vocab_size, embedding_size]
    distances = tf.reduce_sum(distances, axis=-1)  # [..., vocab_size]
    max_distances = tf.reduce_max(distances, axis=-1, keepdims=True)  # [..., 1]
    distances = max_distances - distances  # [..., vocab_size]

    # use sparse ce with logits
    new_targets = tf.zeros(shape=tf.shape(distances)[:-1], dtype=tf.int32)
    out = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=new_targets, logits=distances)

    if self.target_flat is not None:
      #out = self.reduce_func(tf.reduce_mean(out, axis=1))
      out = self.reduce_func(out)
    else:
      out = self.reduce_func(out)

    if self.debug:
      out = tf.Print(out, [out, tf.reduce_sum(target_embeddings)], message="out, Target emb sum: ")

    return out

  @classmethod
  def transform_config_dict(cls, d, network, get_layer):
    super(GeometricCrossEntropy, cls).transform_config_dict(d, network=network, get_layer=get_layer)
    d["target_embedding_layer"] = get_layer(d["target_embedding_layer"])

  def get_error(self):
    # TODO: only get_error when actually needed
    return tf.constant(0.0)

