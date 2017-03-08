require "numo/narray"

module CNTK
module Ops

  module OpsUtil
  class << self

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
      when Numo::NArray, Numeric
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
  end
  end

  module_function
  
  def input_variable(shape, dtype: DataType_Float, needs_grad: false,
                     is_sparse: false, 
                     dynamic_axes: Axis.default_input_variable_dynamic_axes(), 
                     name: '')
    CNTK.__input_variable__(shape, is_sparse, dtype, needs_grad, name, dynamic_axes)
  end
  
  def output_variable(shape: nil, dtype: nil, dynamic_axes: nil, name: "")
    CNTK.__output_variable__(shape, dtype, dynamic_axes, name)
  end

  def placeholder_variable(shape: NDShape.unknown.dimensions(), name: "", dynamic_axes: Axis.unknown_dynamic_axes)
    CNTK.__placeholder_variable__(shape, name, dynamic_axes)
  end

  def constant(*args)
    val = args[0]
    if val.is_a?(Array)
      args[0] = Numo::SFloat[*val]
    end
    Constant.create(*args)
  end

  def parameter(*args) #shape: nil, init_val: nil, dtype: nil, device: nil, name: "")
    Parameter.create(*args)
  end

  def alias(x, name="")
    x = OpsUtil::convert_to_variable( x )
    CNTK.__alias__(x, name)
  end

  def weighted_binary_cross_entropy(output, target, weight, name="")
    output = OpsUtil::convert_to_variable( output )
    target = OpsUtil::convert_to_variable( target )
    weight = OpsUtil::convert_to_variable( weight )
    CNTK.__weighted_binary_cross_entropy__(output, target, weight, name)
  end

  def cross_entropy_with_softmax(output, target, axis=0, name="")
    output = OpsUtil::convert_to_variable( output )
    target = OpsUtil::convert_to_variable( target )
    axis = Axis.from_num(axis)
    CNTK.__cross_entropy_with_softmax__(output, target, axis, name)
  end

  def combine(array, name="")
    a = array.map{|x| OpsUtil::convert_to_variable( x ) }
    CNTK.__combine__(a, name)
  end

  def convolution(kernel: nil, input: nil, strides: [1], sharing: [true],
                  padding: [false], lower_pad: [0], upper_pad: [0],
                  transpose: false, max_temp_mem_size_in_samples: 0, name: "")
    kernel = OpsUtil::convert_to_variable( kernel )
    input  = OpsUtil::convert_to_variable( input  )
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
    x, rois = OpsUtil::convert_to_variable( x, rois )
    CNTK.__roipooling__(x, rois, shape, name)
  end

  def pooling(x, type, shape, strides: [1], padding: [false],
              lower_pad: [0], upper_pad: [0], name: "")
    x = OpsUtil::convert_to_variable( x )
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
    operand, input = OpsUtil::convert_to_variable( operand, input )
    type           = OpsUtil::convert_to_pooling_type( type )
    CNTK.__unpooling__(operand, input, type, shape, strides, padding, lower_pad, upper_pad, name)
  end

  def batch_normalization(x, scale: nil, bias: nil, mean: nil, variance: nil, spatial: false,
                           normalization_time_constant: 5000, blend_time_constant: 0,
                           epsilon: 0.00001, use_cudnn_engine: false, name: "", running_count: 0)
    x,  scale, bias, mean, variance, running_count =
      OpsUtil::convert_to_variable( x, scale, bias, mean, variance, running_count )
    CNTK.__batch_normalization__(x, scale, bias, mean, variance, running_count, spatial,
                                 normalization_time_constant, blend_time_constant,
                                 epsilon, use_cudnn_engine, name)
  end

  def times(left, right, output_rank = 1, name="")
    left, right = OpsUtil::convert_to_variable( left, right)
    CNTK.__times__(left, right, output_rank, name)
  end

  # FIXME
  def lambda_rank(output, gain, group, name="")
    output, gain, group = OpsUtil::convert_to_variable( output, gain, group )
    CNTK.__lambda_rank__(output, gain, group, name)
  end

  # FIXME
  def ndcg_at_1(output, gain, group, name="")
    output, gain, group = OpsUtil::convert_to_variable( output, gain, group )
    CNTK.__ndcgat1__(output, gain, group, name)
  end

  def classification_error(output, target, axis=-1, topN=1, name="")
    output, target = OpsUtil::convert_to_variable( output, target )
    axis   = Axis::from_num(axis)
    CNTK.__classification_error__(output, target, topN, axis, name)
  end

  def transpose(x, axis1=0, axis2=1, name="")
    x = OpsUtil::convert_to_variable( x )
    unless axis1.abs <= x.shape.rank and axis2.abs <= x.shape.rank
      raise ArgumentError, "out of bounds"
    end
    axis1 = Axis::from_num(axis1)
    axis2 = Axis::from_num(axis2)
    CNTK.__transpose_axes__(x, axis1, axis2, name)
  end

  # FIXME
  def edit_distance_error(input_a, input_b, subPen=0, delPen=0, insPen=0,
                          squashInputs=false, samplesToIgnore=[], name='')
    input_a = OpsUtil::convert_to_variable( input_a )
    input_b = OpsUtil::convert_to_variable( input_b )
    CNTK.__edit_distance_error__(input_a, input_b, subPen, delPen, insPen, squashInputs, samplesToIgnore, name)
  end

  (["__negate__", "__sigmoid__", "__tanh__", "__sin__", "__cos__", "__re_lu__",
   "__exp__", "__log__", "__square__", "__sqrt__", "__round__", "__floor__",
   "__ceil__", "__reciprocal__", "__softmax__", "__hardmax__"] ).each{|orig_name|
    mth_name = orig_name.gsub(/_/, "")
    define_method(mth_name) do |*args|
      x    = OpsUtil::convert_to_variable( args[0] )
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
      x, y  = OpsUtil::convert_to_variable( args[0], args[1] )
      name = args[2] || ""
      CNTK.send(orig_name, x, y, name)
    end
  }

end # module Ops
end # module CNTK
