from tensorflow.contrib.framework.python.ops import add_arg_scope
from tensorflow.contrib.framework.python.ops import variables
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import variable_scope

def group_norm(inputs,
               groups=16,
               channels_axis=-1,
               reduction_axes=(-3, -2),
               center=True,
               scale=True,
               epsilon=1e-6,
               activation_fn=None,
               param_initializers=None,
               reuse=None,
               variables_collections=None,
               outputs_collections=None,
               trainable=True,
               scope=None,
               mean_close_to_zero=False):
    """Functional interface for the group normalization layer.
  Reference: https://arxiv.org/abs/1803.08494.
    "Group Normalization", Yuxin Wu, Kaiming He
  Args:
    inputs: A Tensor with at least 2 dimensions one which is channels. All
     shape dimensions except for batch must be fully defined.
    groups: Integer. Divide the channels into this number of groups over which
      normalization statistics are computed. This number must be commensurate
      with the number of channels in `inputs`.
    channels_axis: An integer. Specifies index of channels axis which will be
      broken into `groups`, each of which whose statistics will be computed
      across. Must be mutually exclusive with `reduction_axes`. Preferred usage
      is to specify negative integers to be agnostic as to whether a batch
      dimension is included.
    reduction_axes: Tuple of integers. Specifies dimensions over which
       statistics will be accumulated. Must be mutually exclusive with
       `channels_axis`. Statistics will not be accumulated across axes not
       specified in `reduction_axes` nor `channel_axis`. Preferred usage is to
       specify negative integers to be agnostic to whether a batch dimension is
       included.
      Some sample usage cases:
        NHWC format: channels_axis=-1, reduction_axes=[-3, -2]
        NCHW format: channels_axis=-3, reduction_axes=[-2, -1]
    center: If True, add offset of `beta` to normalized tensor. If False, `beta`
      is ignored.
    scale: If True, multiply by `gamma`. If False, `gamma` is
      not used. When the next layer is linear (also e.g. `nn.relu`), this can be
      disabled since the scaling can be done by the next layer.
    epsilon: Small float added to variance to avoid dividing by zero.
    activation_fn: Activation function, default set to None to skip it and
      maintain a linear activation.
    param_initializers: Optional initializers for beta, gamma, moving mean and
      moving variance.
    reuse: Whether or not the layer and its variables should be reused. To be
      able to reuse the layer scope must be given.
    variables_collections: Optional collections for the variables.
    outputs_collections: Collections to add the outputs.
    trainable: If `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    scope: Optional scope for `variable_scope`.
    mean_close_to_zero: The mean of `input` before ReLU will be close to zero
      when batch size >= 4k for Resnet-50 on TPU. If `True`, use
      `nn.sufficient_statistics` and `nn.normalize_moments` to calculate the
      variance. This is the same behavior as `fused` equals `True` in batch
      normalization. If `False`, use `nn.moments` to calculate the variance.
      When `mean` is close to zero, like 1e-4, use `mean` to calculate the
      variance may have poor result due to repeated roundoff error and
      denormalization in `mean`.  When `mean` is large, like 1e2,
      sum(`input`^2) is so large that only the high-order digits of the elements
      are being accumulated. Thus, use sum(`input` - `mean`)^2/n to calculate
      the variance has better accuracy compared to (sum(`input`^2)/n - `mean`^2)
      when `mean` is large.
  Returns:
    A `Tensor` representing the output of the operation.
  Raises:
    ValueError: If the rank of `inputs` is undefined.
    ValueError: If rank or channels dimension of `inputs` is undefined.
    ValueError: If number of groups is not commensurate with number of channels.
    ValueError: If reduction_axes or channels_axis are out of bounds.
    ValueError: If reduction_axes are not mutually exclusive with channels_axis.
    """
    # TODO(shlens): Support partially defined shapes for the inputs.

    inputs = ops.convert_to_tensor(inputs)

    if inputs.shape.ndims is None:
        raise ValueError('Inputs %s has undefined rank.' % inputs.name)
    if channels_axis > (inputs.shape.ndims - 1):
        raise ValueError('Axis is out of bounds.')

  # Use dynamic shape for not fully defined dimensions in the inputs.
    dyanmic_shape = array_ops.shape(inputs)
    input_shape_list = []
    for i, dim in enumerate(inputs.shape.as_list()):
        if dim is None:
            input_shape_list.append(dyanmic_shape[i])
        else:
            input_shape_list.append(dim)
    # print('{}, {}'.format(scope, input_shape_list))
  # Standardize the channels_axis to be positive and identify # of channels.
    if channels_axis < 0:
        channels_axis = inputs.shape.ndims + channels_axis
    channels = inputs.shape[channels_axis].value

    if channels is None:
        raise ValueError('Inputs %s has undefined channel dimension: %d.' % (
        inputs.name, channels_axis))

  # Standardize the reduction_axes to be positive.
    reduction_axes = list(reduction_axes)
    for i in range(len(reduction_axes)):
        if reduction_axes[i] < 0:
            reduction_axes[i] += inputs.shape.ndims

    for a in reduction_axes:
        if a > inputs.shape.ndims:
            raise ValueError('Axis is out of bounds.')
        if inputs.shape[a].value is None:
            raise ValueError('Inputs %s has undefined dimensions %d.' % (
                inputs.name, a))
        if channels_axis == a:
            raise ValueError('reduction_axis must be mutually exclusive '
                       'with channels_axis')
    if groups > channels:
        raise ValueError('Invalid groups %d for %d channels.' % (groups, channels))
    if channels % groups != 0:
        raise ValueError('%d channels is not commensurate with %d groups.' %
                     (channels, groups))

  # Determine axes before channels. Some examples of common image formats:
  #  'NCHW': before = [N], after = [HW]
  #  'NHWC': before = [NHW], after = []
    axes_before_channels = input_shape_list[:channels_axis]
    axes_after_channels = input_shape_list[channels_axis+1:]

    # Manually broadcast the parameters to conform to the number of groups.
    params_shape_broadcast = ([1] * len(axes_before_channels) +
                                [groups, channels // groups] +
                                [1] * len(axes_after_channels))

    # Reshape the input by the group within the channel dimension.
    inputs_shape = (axes_before_channels + [groups, channels // groups] +
                    axes_after_channels)
    #   print(inputs_shape)
    inputs = array_ops.reshape(inputs, inputs_shape)

    # Determine the dimensions across which moments are calculated.
    moments_axes = [channels_axis + 1]
    for a in reduction_axes:
        if a > channels_axis:
            moments_axes.append(a + 1)
        else:
            moments_axes.append(a)

    with variable_scope.variable_scope(
        scope, 'GroupNorm', [inputs], reuse=reuse) as sc:
    # Note that the params_shape is the number of channels always.
        params_shape = [channels]

    # Allocate parameters for the beta and gamma of the normalization.
        beta, gamma = None, None
        dtype = inputs.dtype.base_dtype
        if param_initializers is None:
            param_initializers = {}
        if center:
            beta_collections = utils.get_variable_collections(
                variables_collections, 'beta')
            beta_initializer = param_initializers.get(
                'beta', init_ops.zeros_initializer())
            beta = variables.model_variable('beta',
                                        shape=params_shape,
                                        dtype=dtype,
                                        initializer=beta_initializer,
                                        collections=beta_collections,
                                        trainable=trainable)
            beta = array_ops.reshape(beta, params_shape_broadcast)

        if scale:
            gamma_collections = utils.get_variable_collections(
                variables_collections, 'gamma')
            gamma_initializer = param_initializers.get(
                'gamma', init_ops.ones_initializer())
            gamma = variables.model_variable('gamma',
                                        shape=params_shape,
                                        dtype=dtype,
                                        initializer=gamma_initializer,
                                        collections=gamma_collections,
                                        trainable=trainable)
            gamma = array_ops.reshape(gamma, params_shape_broadcast)

        # Calculate the moments.
        if mean_close_to_zero:
        # One pass algorithm returns better result when mean is close to zero.
            counts, means_ss, variance_ss, _ = nn.sufficient_statistics(
                inputs, moments_axes, keep_dims=True)
            mean, variance = nn.normalize_moments(
                counts, means_ss, variance_ss, shift=None)
        else:
            mean, variance = nn.moments(inputs, moments_axes, keep_dims=True)

        # Compute normalization.
        # TODO(shlens): Fix nn.batch_normalization to handle the 5-D Tensor
        # appropriately so that this operation may be faster.
        gain = math_ops.rsqrt(variance + epsilon)
        offset = -mean * gain
        if gamma is not None:
            gain *= gamma
            offset *= gamma
        if beta is not None:
            offset += beta
        
        outputs = inputs * gain + offset

        # Collapse the groups into the channel dimension.
        outputs = array_ops.reshape(outputs, input_shape_list)

        if activation_fn is not None:
            outputs = activation_fn(outputs)
        return utils.collect_named_outputs(outputs_collections, sc.name, outputs)
   



