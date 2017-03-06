require "numo/narray"

module CNTK
module Ops

  module OpsUtil
  class << self
    def convert_to_variable(x, dtype=Numo::SFloat)
      case x
      when Variable
        x
      when Numo::NArray
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

  def convolution(kernel: nil, input: nil,  strides: [1], sharing: [true],
                  padding: [false], lower_pad: [0], upper_pad: [0],
                  transpose: false, max_temp_mem_size_in_samples: 0, name: "")
    kernel = OpsUtil::convert_to_variable(kernel)
    input  = OpsUtil::convert_to_variable(input)
    CNTK.__convolution__(kernel, input, strides, sharing, padding, lower_pad, upper_pad,
                         transpose, max_temp_mem_size_in_samples, name)
  end

  def lambda_rank(output, gain, group, name="")
    output = OpsUtil::convert_to_variable( output )
    gain   = OpsUtil::convert_to_variable( gain   )
    group  = OpsUtil::convert_to_variable( group  )
    CNTK.__lambda_rank__(output, gain, group, name)
  end

  def ndcg_at_1(output, gain, group, name="")
    output = OpsUtil::convert_to_variable( output )
    gain   = OpsUtil::convert_to_variable( gain   )
    group  = OpsUtil::convert_to_variable( group  )
    CNTK.__ndcgat1__(output, gain, group, name)
  end

  def classification_error(output, target, axis=-1, topN=1, name="")
    output = OpsUtil::convert_to_variable( output )
    target = OpsUtil::convert_to_variable( target )
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
      x    = OpsUtil::convert_to_variable( args[0] )
      y    = OpsUtil::convert_to_variable( args[1] )
      name = args[2] || ""
      CNTK.send(orig_name, x, y, name)
    end
  }

end # module Ops
end # module CNTK
