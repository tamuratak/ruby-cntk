module CNTK
class Trainer

  # @param args [Hash<Variable,MinibatchData>]
  # @option opt [Array<Variable>] :outputs
  # @option opt [DeviceDescriptor] :device
  def train_minibatch(args, outputs: nil, device: DeviceDescriptor.use_default_device)
    if outputs
      out = StdUMapVariableValue.new()
      outputs.each{|out_var|
        # By setting nullptr, Forward function implemented in C++ will allocate Value object with required storage.
        out.__set_nullptr__(out_var)
      }
      updated = __train_minibatchdata__(args, out, device)
      return [updated, out]
    else
      __train_minibatchdata__(args, device)
    end
  end


class << self

  def create(model: nil, loss: nil, evaluation: nil, learners: nil)
    unless model and loss and learners
      raise ArgumentError, "model, loss function, and learners needed"
    end
    model     = variable_to_function(model)
    loss      = variable_to_function(loss)
    evaluation = variable_to_function(evaluation) if evaluation
    learners  = [learners] unless learners.is_a?(Array)
    if evaluation
      CNTK.__create_trainer__(model, loss, evaluation, learners)
    else
      CNTK.__create_trainer__(model, loss, learners)
    end
  end

  private

  def variable_to_function(x)
    case x
    when Function
      x
    when Variable
      CNTK::Ops.combine([x])
    else
      raise ArgumentError
    end
  end

end # class << self
end # class Trainer
end # module CNTK
