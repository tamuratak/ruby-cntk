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
        if x.outputs.size == 1
          x.output
        else
          raise ArgumentError, "the output size of Function expected to be 1"
        end
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
  def input_variable(shape, dtype: DataType_Float, needs_grad: false,
                     is_sparse: false, 
                     dynamic_axes: nil,
                     name: '')
    if dynamic_axes
      dynamic_axes = dynamic_axes.reverse
    else
      dynamic_axes = Axis.default_input_variable_dynamic_axes()
    end
    CNTK.__input_variable__(shape, is_sparse, dtype, needs_grad, name, dynamic_axes)
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
                  transpose: false, max_temp_mem_size_in_samples: 0, name: "")
    kernel = Ops.convert_to_variable( kernel )
    input  = Ops.convert_to_variable( input  )
    CNTK.__convolution__(kernel, input, strides, sharing, padding, lower_pad, upper_pad,
                         transpose, max_temp_mem_size_in_samples, name)
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
              lower_pad: [0], upper_pad: [0], name: "")
    x = Ops.convert_to_variable( x )
    case type
    when :max
      type = CNTK::PoolingType_Max
    when :average
      type = CNTK::PoolingType_Average
    else
      raise ArgumentError, "unknown pooling type"
    end
    CNTK.__pooling__(x, type, shape, strides, padding, lower_pad, upper_pad, name)
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

  def times(left, right, output_rank = 1, name="")
    left, right = Ops.convert_to_variable( left, right )
    # change the order because CNTK a column-major.
    CNTK.__times__(right, left, output_rank, name)
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

  (["__negate__", "__sigmoid__", "__tanh__", "__sin__", "__cos__", "__re_lu__",
   "__exp__", "__log__", "__square__", "__sqrt__", "__round__", "__floor__",
   "__ceil__", "__reciprocal__", "__softmax__", "__hardmax__"] ).each{|orig_name|
    mth_name = orig_name.gsub(/_/, "")
    define_method(mth_name) do |*args|
      x    = Ops.convert_to_variable( args[0] )
      name = args[1] || ""
      CNTK.send(orig_name, x, name)
    end
  }

  (["__plus__", "__minus__", "__log_add_exp__", "abs", "__element_times__",
   "__element_divide__", "__equal__", "__not_equal__", "__less__", "__less_equal__",
   "__greater__", "__greater_equal__"] +
   ["__cosine_distance__", "__binary_cross_entropy__", "__squared_error__"]).each{|orig_name|
    mth_name = orig_name.gsub(/__/, "")
    define_method(mth_name) do |*args|
      x, y  = Ops.convert_to_variable( args[0], args[1] )
      name = args[2] || ""
      CNTK.send(orig_name, x, y, name)
    end
  }

end # module Ops
end # module CNTK
