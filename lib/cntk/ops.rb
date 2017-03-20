require "numo/narray"

module CNTK
module Ops

  class << self

    def reverse_reshape_axis(axis)
      axis = Axis.new(axis) if axis.is_a?(Numeric)
      case
      when axis.is_static_axis
        axis 
      when axis == Axis.end_static_axis
        Axis.new(0)
      when axis == Axis.new(0)
        Axis.end_static_axis
      else
        Axis(-axis.static_axis_index)
      end
    end

    def reverse_dynamic_axes(axes)
      axes = [axes] unless axes.is_a?(Array)
      axes.each{|ax|
        raise ArgumentError, "Axis expected" unless ax.is_a?(Axis)
      }
      axes.reverse
    end

    def convert_to_pooling_type(type)
      case type
      when :max
        type = CNTK::PoolingType_Max
      when :average
        type = CNTK::PoolingType_Average
      else
        raise ArgumentError, "unknown pooling type"
      end
    end

    def convert_to_variable(*vars)
      if vars.size == 1
        convert_to_one_variable(vars[0])
      else
        dtype = highest_precision_type(*vars)
        return vars.map{|v| convert_to_one_variable(v, dtype) }
      end
    end

    def convert_to_one_variable(x, dtype = Numo::SFloat)
      case x
      when Variable
        x
      when Function
        x.output
      when Value, Numo::NArray, Numeric
        Ops.constant(x)
      when Array
        Ops.constant( dtype[*x] )
      else
        raise ArgumentError, "CNTK::Variable, Numo::NArray, or Array expected"
      end
    end

    def highest_precision_type(*args)
      types = args.map{|v|
        case v
        when Variable
          Variable::DataType[v.get_data_type]
        when Numo::NArray
          v.class
        else
          nil
        end
      }
      if types.include?(Numo::DFloat)
        Numo::DFloat
      else
        Numo::SFloat
      end
    end
  end   # class << self

  module_function
  
  #
  # variable ops
  #
  def input_variable(shape, dtype: DataType_Float, needs_gradient: false,
                     is_sparse: false, 
                     dynamic_axes: nil,
                     name: '')
    if dynamic_axes
      dynamic_axes = dynamic_axes.reverse
    else
      dynamic_axes = Axis.default_input_variable_dynamic_axes()
    end
    CNTK.__input_variable__(shape, is_sparse, dtype, needs_gradient, name, dynamic_axes)
  end
  
  def output_variable(shape: nil, dtype: nil, dynamic_axes: nil, name: "")
    if dynamic_axes
      dynamic_axes = dynamic_axes.reverse
    end
    CNTK.__output_variable__(shape, dtype, dynamic_axes, name)
  end

  def placeholder_variable(shape: NDShape.unknown.dimensions(), name: "", dynamic_axes: nil)
    if dynamic_axes
      dynamic_axes = dynamic_axes.reverse
    else
      dynamic_axes = Axis.unknown_dynamic_axes
    end
    CNTK.__placeholder_variable__(shape, name, dynamic_axes)
  end

  def constant(*args)
    val = args[0]
    if val.is_a?(Array)
      args[0] = Numo::SFloat[*val]
    end
    Constant.create(*args)
  end

  def parameter(*args)
    Parameter.create(*args)
  end

  #
  # ops
  #
  def alias(x, name="")
    x = Ops.convert_to_variable( x )
    CNTK.__alias__(x, name)
  end

  def weighted_binary_cross_entropy(output, target, weight, name="")
    output = Ops.convert_to_variable( output )
    target = Ops.convert_to_variable( target )
    weight = Ops.convert_to_variable( weight )
    CNTK.__weighted_binary_cross_entropy__(output, target, weight, name)
  end

  def cross_entropy_with_softmax(output, target, axis=0, name="")
    output = Ops.convert_to_variable( output )
    target = Ops.convert_to_variable( target )
    axis = Axis.from_num(axis)
    CNTK.__cross_entropy_with_softmax__(output, target, axis, name)
  end

  def combine(array, name="")
    a = array.map{|x| Ops.convert_to_variable( x ) }
    CNTK.__combine__(a, name)
  end

  def convolution(kernel: nil, input: nil, strides: [1], sharing: [true],
                  padding: [false], lower_pad: [0], upper_pad: [0],
                  max_temp_mem_size_in_samples: 0, name: "")
    kernel = Ops.convert_to_variable( kernel )
    input  = Ops.convert_to_variable( input  )
    CNTK.__convolution__(kernel, input, strides, sharing, padding, lower_pad, upper_pad,
                         max_temp_mem_size_in_samples, name)
  end


  # CNTK's NDArray is column-major.
  # So to specify rois, remember it.
  #          y
  #     __________
  #     |
  #    x|
  #     |
  def roipooling(x, rois, shape, name="")
    x, rois = Ops.convert_to_variable( x, rois )
    CNTK.__roipooling__(x, rois, shape, name)
  end

  def pooling(x, type, shape, strides: [1], padding: [false],
              lower_pad: [0], upper_pad: [0], ceil_out_dim: false, name: "")
    x = Ops.convert_to_variable( x )
    case type
    when :max
      type = CNTK::PoolingType_Max
    when :average
      type = CNTK::PoolingType_Average
    else
      raise ArgumentError, "unknown pooling type"
    end
    CNTK.__pooling__(x, type, shape, strides, padding, lower_pad, upper_pad, ceil_out_dim, name)
  end

  def unpooling(operand, input, type, shape, strides: [1], padding: [false],
                lower_pad: [0], upper_pad: [0], name: "")
    operand, input = Ops.convert_to_variable( operand, input )
    type           = Ops.convert_to_pooling_type( type )
    CNTK.__unpooling__(operand, input, type, shape, strides, padding, lower_pad, upper_pad, name)
  end

  def batch_normalization(x, scale: nil, bias: nil, mean: nil, variance: nil, spatial: false,
                           normalization_time_constant: 5000, blend_time_constant: 0,
                           epsilon: 0.00001, use_cudnn_engine: false, name: "", running_count: 0)
    x,  scale, bias, mean, variance, running_count =
      Ops.convert_to_variable( x, scale, bias, mean, variance, running_count )
    CNTK.__batch_normalization__(x, scale, bias, mean, variance, running_count, spatial,
                                 normalization_time_constant, blend_time_constant,
                                 epsilon, use_cudnn_engine, name)
  end

  def times(left, right, output_rank: 1, infer_input_rank_to_map: -1, name: "")
    left, right = Ops.convert_to_variable( left, right )
    # change the order because CNTK a column-major.
    CNTK.__times__(right, left, output_rank, infer_input_rank_to_map, name)
  end

  def times_transpose(left, right, output_rank = 1, name="")
    left, right = Ops.convert_to_variable( left, right )
    CNTK.__transpose_times__(right, left, output_rank, name="")
  end

  def clip(x, min, max, name="")
    x, min, max = Ops.convert_to_variable( x, min, max )
    CNTK.__clip__(x, min, max, name)
  end

  def element_select(x, if_true, if_else, name="")
    x, if_true, if_else = Ops.convert_to_variable( x, if_true, if_else )
    CNTK.__element_select__(x, if_true, if_else, name)
  end

  def future_value(x, init=0, time_step=1, name="")
    x, init = Ops.convert_to_variable( x, init )
    CNTK.__future_value__(x, init, time_step, name)
  end

  def past_value(x, init=0, time_step=1, name="")
    x, init = Ops.convert_to_variable( x, init )
    CNTK.__past_value__(x, init, time_step, name)
  end

  def reshape(x, shape, begin_axis=Axis.new(0), end_axis=Axis.end_static_axis(), name="")
    x = Ops.convert_to_variable( x )
    begin_axis = Ops.reverse_reshape_axis(begin_axis)
    end_axis   = Ops.reverse_reshape_axis(end_axis  )
    CNTK.__reshape__(x, shape, begin_axis, end_axis, name)
  end

  def transpose(x, axis1=0, axis2=1, name="")
    x = Ops.convert_to_variable( x )
    unless axis1.abs <= x.shape.rank and axis2.abs <= x.shape.rank
      raise ArgumentError, "out of bounds"
    end
    axis1 = Axis.from_num(axis1)
    axis2 = Axis.from_num(axis2)
    CNTK.__transpose_axes__(x, axis1, axis2, name)
  end

  def slice(x, axis, begin_index, end_index, name="")
    x = Ops.convert_to_variable( x )
    axis = Axis.from_num(axis)
    CNTK.__slice__(x, axis, begin_index, end_index, name)
  end

  def splice(x, axis=-1, name="")
    x = x.map{|var| Ops.convert_to_variable( var ) }
    axis = Axis.from_num(axis)
    CNTK.__splice__(x, axis, name)
  end

  def reduce_sum(x, axis=nil, name="")
    x = Ops.convert_to_variable( x )
    axis = Axis.from_num(axis)
    CNTK.__reduce_sum__(x, axis, name)
  end

  def reduce_log_sum_exp(x, axis=nil, name="")
    x = Ops.convert_to_variable( x )
    axis = Axis.from_num(axis)
    CNTK.__reduce_log_sum__(x, axis, name)
  end

  def reduce_mean(x, axis=nil, name="")
    x = Ops.convert_to_variable( x )
    axis = Axis.from_num(axis)
    CNTK.__reduce_mean__(x, axis, name)
  end

  def reduce_max(x, axis=nil, name="")
    x = Ops.convert_to_variable( x )
    axis = Axis.from_num(axis)
    CNTK.__reduce_max__(x, axis, name)
  end

  def reduce_min(x, axis=nil, name="")
    x = Ops.convert_to_variable( x )
    axis = Axis.from_num(axis)
    CNTK.__reduce_min__(x, axis, name)
  end

  def reduce_prod(x, axis=nil, name="")
    x = Ops.convert_to_variable( x )
    axis = Axis.from_num(axis)
    CNTK.__reduce_prod__(x, axis, name)
  end

  def random_sample(weights, num_samples, allow_dup, name="")
    weights = Ops.convert_to_variable( weights )
    CNTK.__random_sample__(weights, num_samples, allow_dup, name)
  end

  def random_sample_inclusion_frequency(weights, num_samples, allow_dup, name="")
    weights = Ops.convert_to_variable( weights )
    CNTK.__random_sample_inclusion_frequency__(weights, num_samples, allow_dup, name)
  end

  def dropout(x, rate=0.0, name="")
    if rate < 0 or rate >= 1
      raise ArgumentError, "dropout_rate must be in the interval [0,1)"
    end
    x = Ops.convert_to_variable( x )
    CNTK.__dropout__(x, rate, name)
  end

  # FIXME
  def lambda_rank(output, gain, group, name="")
    output, gain, group = Ops.convert_to_variable( output, gain, group )
    CNTK.__lambda_rank__(output, gain, group, name)
  end

  # FIXME
  def ndcg_at_1(output, gain, group, name="")
    output, gain, group = Ops.convert_to_variable( output, gain, group )
    CNTK.__ndcgat1__(output, gain, group, name)
  end

  def classification_error(output, target, axis=-1, topN=1, name="")
    output, target = Ops.convert_to_variable( output, target )
    axis   = Axis::from_num(axis)
    CNTK.__classification_error__(output, target, topN, axis, name)
  end

  # FIXME
  def edit_distance_error(input_a, input_b, subPen=0, delPen=0, insPen=0,
                          squashInputs=false, samplesToIgnore=[], name='')
    input_a = Ops.convert_to_variable( input_a )
    input_b = Ops.convert_to_variable( input_b )
    CNTK.__edit_distance_error__(input_a, input_b, subPen, delPen, insPen, squashInputs, samplesToIgnore, name)
  end

  def negate(x=nil, name: "")
    x = x || Ops.placeholder_variable(name: "x")
    x = Ops.convert_to_variable( x )
    CNTK.__negate__(x, name)
  end

  def sigmoid(x=nil, name: "")
    x = x || Ops.placeholder_variable(name: "x")
    x = Ops.convert_to_variable( x )
    CNTK.__sigmoid__(x, name)
  end

  def tanh(x=nil, name: "")
    x = x || Ops.placeholder_variable(name: "x")
    x = Ops.convert_to_variable( x )
    CNTK.__tanh__(x, name)
  end

  def sin(x=nil, name: "")
    x = x || Ops.placeholder_variable(name: "x")
    x = Ops.convert_to_variable( x )
    CNTK.__sin__(x, name)
  end

  def cos(x=nil, name: "")
    x = x || Ops.placeholder_variable(name: "x")
    x = Ops.convert_to_variable( x )
    CNTK.__cos__(x, name)
  end

  def relu(x=nil, name: "")
    x = x || Ops.placeholder_variable(name: "x")
    x = Ops.convert_to_variable( x )
    CNTK.__re_lu__(x, name)
  end

  def exp(x=nil, name: "")
    x = x || Ops.placeholder_variable(name: "x")
    x = Ops.convert_to_variable( x )
    CNTK.__exp__(x, name)
  end

  def log(x=nil, name: "")
    x = x || Ops.placeholder_variable(name: "x")
    x = Ops.convert_to_variable( x )
    CNTK.__log__(x, name)
  end

  def square(x=nil, name: "")
    x = x || Ops.placeholder_variable(name: "x")
    x = Ops.convert_to_variable( x )
    CNTK.__square__(x, name)
  end

  def sqrt(x=nil, name: "")
    x = x || Ops.placeholder_variable(name: "x")
    x = Ops.convert_to_variable( x )
    CNTK.__sqrt__(x, name)
  end

  def round(x=nil, name: "")
    x = x || Ops.placeholder_variable(name: "x")
    x = Ops.convert_to_variable( x )
    CNTK.__round__(x, name)
  end

  def floor(x=nil, name: "")
    x = x || Ops.placeholder_variable(name: "x")
    x = Ops.convert_to_variable( x )
    CNTK.__floor__(x, name)
  end

  def ceil(x=nil, name: "")
    x = x || Ops.placeholder_variable(name: "x")
    x = Ops.convert_to_variable( x )
    CNTK.__ceil__(x, name)
  end

  def reciprocal(x=nil, name: "")
    x = x || Ops.placeholder_variable(name: "x")
    x = Ops.convert_to_variable( x )
    CNTK.__reciprocal__(x, name)
  end

  def softmax(x=nil, name: "")
    x = x || Ops.placeholder_variable(name: "x")
    x = Ops.convert_to_variable( x )
    CNTK.__softmax__(x, name)
  end

  def hardmax(x=nil, name: "")
    x = x || Ops.placeholder_variable(name: "x")
    x = Ops.convert_to_variable( x )
    CNTK.__hardmax__(x, name)
  end

  def plus(x=nil, y=nil, name: "")
    x = x || Ops.placeholder_variable(name: "x")
    y = y || Ops.placeholder_variable(name: "y")
    x, y = Ops.convert_to_variable( x, y )
    CNTK.__plus__(x, y, name)
  end

  def minus(x=nil, y=nil, name: "")
    x = x || Ops.placeholder_variable(name: "x")
    y = y || Ops.placeholder_variable(name: "y")
    x, y = Ops.convert_to_variable( x, y )
    CNTK.__minus__(x, y, name)
  end

  def log_add_exp(x=nil, y=nil, name: "")
    x = x || Ops.placeholder_variable(name: "x")
    y = y || Ops.placeholder_variable(name: "y")
    x, y = Ops.convert_to_variable( x, y )
    CNTK.__log_add_exp__(x, y, name)
  end

  def abs(x=nil, y=nil, name: "")
    x = x || Ops.placeholder_variable(name: "x")
    y = y || Ops.placeholder_variable(name: "y")
    x, y = Ops.convert_to_variable( x, y )
    CNTK.abs(x, y, name)
  end

  def element_times(x=nil, y=nil, name: "")
    x = x || Ops.placeholder_variable(name: "x")
    y = y || Ops.placeholder_variable(name: "y")
    x, y = Ops.convert_to_variable( x, y )
    CNTK.__element_times__(x, y, name)
  end

  def element_divide(x=nil, y=nil, name: "")
    x = x || Ops.placeholder_variable(name: "x")
    y = y || Ops.placeholder_variable(name: "y")
    x, y = Ops.convert_to_variable( x, y )
    CNTK.__element_divide__(x, y, name)
  end

  def equal(x=nil, y=nil, name: "")
    x = x || Ops.placeholder_variable(name: "x")
    y = y || Ops.placeholder_variable(name: "y")
    x, y = Ops.convert_to_variable( x, y )
    CNTK.__equal__(x, y, name)
  end

  def not_equal(x=nil, y=nil, name: "")
    x = x || Ops.placeholder_variable(name: "x")
    y = y || Ops.placeholder_variable(name: "y")
    x, y = Ops.convert_to_variable( x, y )
    CNTK.__not_equal__(x, y, name)
  end

  def less(x=nil, y=nil, name: "")
    x = x || Ops.placeholder_variable(name: "x")
    y = y || Ops.placeholder_variable(name: "y")
    x, y = Ops.convert_to_variable( x, y )
    CNTK.__less__(x, y, name)
  end

  def less_equal(x=nil, y=nil, name: "")
    x = x || Ops.placeholder_variable(name: "x")
    y = y || Ops.placeholder_variable(name: "y")
    x, y = Ops.convert_to_variable( x, y )
    CNTK.__less_equal__(x, y, name)
  end

  def greater(x=nil, y=nil, name: "")
    x = x || Ops.placeholder_variable(name: "x")
    y = y || Ops.placeholder_variable(name: "y")
    x, y = Ops.convert_to_variable( x, y )
    CNTK.__greater__(x, y, name)
  end

  def greater_equal(x=nil, y=nil, name: "")
    x = x || Ops.placeholder_variable(name: "x")
    y = y || Ops.placeholder_variable(name: "y")
    x, y = Ops.convert_to_variable( x, y )
    CNTK.__greater_equal__(x, y, name)
  end

  def cosine_distance(x=nil, y=nil, name: "")
    x = x || Ops.placeholder_variable(name: "x")
    y = y || Ops.placeholder_variable(name: "y")
    x, y = Ops.convert_to_variable( x, y )
    CNTK.__cosine_distance__(x, y, name)
  end

  def binary_cross_entropy(x=nil, y=nil, name: "")
    x = x || Ops.placeholder_variable(name: "x")
    y = y || Ops.placeholder_variable(name: "y")
    x, y = Ops.convert_to_variable( x, y )
    CNTK.__binary_cross_entropy__(x, y, name)
  end

  def squared_error(x=nil, y=nil, name: "")
    x = x || Ops.placeholder_variable(name: "x")
    y = y || Ops.placeholder_variable(name: "y")
    x, y = Ops.convert_to_variable( x, y )
    CNTK.__squared_error__(x, y, name)
  end


end # module Ops
end # module CNTK
