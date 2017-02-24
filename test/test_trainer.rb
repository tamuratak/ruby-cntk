require "test/unit"
require "cntk"
require "numo/narray"

class TestTrain < Test::Unit::TestCase
  include CNTK
  include CNTK::Ops

  def test_learner
    a = parameter(shape: [3], init: 1)
    x = input_variable([3])
    z = a * x
    z.parameters
    sch = MomentumSchedule.new(0.1, MomentumSchedule::UnitType_Sample)
    CNTK.sgd_learner(z.parameters, sch)
  end

  def test_learner_2
    CNTK.training_parameter_schedule(1, :minibatch)
    CNTK.training_parameter_schedule([1,2,3], :minibatch)
    CNTK.training_parameter_schedule([1,2,3], :minibatch, 100)

    CNTK.momentum_schedule(1)
  end

  def test_learner_3
    AdditionalLearningOptions.new
  end
end
