module CNTK
  module_function

  def input_variable(shape, dtype: DataType_Float, needs_grad: false,
                     is_sparse: false, 
                     dynamic_axes: Axis.default_input_variable_dynamic_axes(), 
                     name: '')

    __input_variable__(shape, is_sparse, dtype, needs_grad, name, dynamic_axes)

  end

  def placeholder_variable(shape, name: "", dynamic_axes: Axis.unknown_dynamic_axes)
    __placeholder_variable__(shape, name, dynamic_axes)
  end

  def constant(*args)
    Constant.create(*args)
  end

  def parameter(*args) #shape: nil, init_val: nil, dtype: nil, device: nil, name: "")
    Parameter.create(*args)
  end

end
