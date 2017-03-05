module CNTK
class Trainer
class << self

  def create(model: nil, loss: nil, evalation: nil, learners: nil)
    unless model and loss and learners
      raise ArgumentError, "model, loss function and learners needed"
    end
    model     = variable_to_function(model)
    loss      = variable_to_function(loss)
    evalation = variable_to_function(evalation) if evalation
    if evalation
      CNTK.__create_trainer__(model, loss, evalation, learners)
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

end
end
end
