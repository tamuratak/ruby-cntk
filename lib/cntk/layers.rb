module CNTK
module Layers

  module_function

  # @return [Function]
  def dense(output_shape, init: Initializer.glorot_uniform,
            input_shape: [CNTK::NDShape::Inferred_dimension],
            use_bias: true, init_bias: 0, name: "")
    _W = Ops.parameter(shape: input_shape + output_shape, init: init,      name: "W")
     b = Ops.parameter(shape:               output_shape, init: init_bias, name: "b")
     x = Ops.placeholder_variable(name: "x")
     Ops.times(x, _W, output_rank: output_shape.size, infer_input_rank_to_map: 0) + b
  end

end
end
