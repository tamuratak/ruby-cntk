module CNTK
module Ops

  module_function
  
  def input_variable(shape, dtype: DataType_Float, needs_grad: false,
                     is_sparse: false, 
                     dynamic_axes: Axis.default_input_variable_dynamic_axes(), 
                     name: '')
    
    __input_variable__(shape, is_sparse, dtype, needs_grad, name, dynamic_axes)
    
  end
  
  def output_variable(shape: nil, dtype: nil, dynamic_axes: nil, name: "")
    __output_variable__(shape, dtype, dynamic_axes, name)
  end

  def placeholder_variable(shape: NDShape.unknown.dimensions(), name: "", dynamic_axes: Axis.unknown_dynamic_axes)
    __placeholder_variable__(shape, name, dynamic_axes)
  end

  def constant(*args)
    Constant.create(*args)
  end

  def parameter(*args) #shape: nil, init_val: nil, dtype: nil, device: nil, name: "")
    Parameter.create(*args)
  end

  def sin(x = nil, name: "")
    if x
      CNTK.__sin__(x, name)
    else
      x = placeholder_variable()
      CNTK.__sin__(x, name)
    end
  end

end
end
