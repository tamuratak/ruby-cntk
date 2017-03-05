module CNTK
module Ops

  module OpsUtil
    def self.convert_to_variable(x)
      case x
      when Variable
        x
      when Numo::NArray, Array
        Ops.constant(x)
      else
        raise ArgumentError, "CNTK::Variable, Numo::NArray, or Array expected"
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
    args
    val = args[0]
    if val.is_a?(Array)
      args[0] = Numo::DFloat[*val]
    end
    Constant.create(*args)
  end

  def parameter(*args) #shape: nil, init_val: nil, dtype: nil, device: nil, name: "")
    Parameter.create(*args)
  end

  ["__negate__", "__sigmoid__", "__tanh__", "__sin__", "__cos__", "__re_lu__",
   "__exp__", "__log__", "__square__", "__sqrt__", "__round__", "__floor__",
   "__ceil__", "__reciprocal__", "__softmax__", "__hardmax__"].each{|orig_name|
    mth_name = orig_name.gsub(/_/, "")
    define_method(mth_name) do |*args|
      x    = args[0] or raise ArgumentError
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
      x    = args[0] or raise ArgumentError
      y    = args[1] or raise ArgumentError
      name = args[2] || ""
      CNTK.send(orig_name, x, y, name)
    end
  }

  def transpose(x, axis1=0, axis2=1, name="")

  end

end # module Ops
end # module CNTK
