require "test/unit"
require "cntk"
require "numo/narray"

class TestTrain < Test::Unit::TestCase
  include CNTK
  include CNTK::Ops

  def test_learner
    a = parameter(shape: [3], init: 1)
    x = input_variable([3])
    y = a * x
    sch = CNTK::Learner.training_parameter_schedule(1, :minibatch)
    learner = CNTK::Learner.sgd(y.parameters, sch)
#    puts (CNTK.methods - Object.methods).map{|s| s.to_s.dump }.join(", ")
    z = input_variable([1])
    CNTK::Trainer::create(model: y, loss: z, learners: [learner])
  end

  def test_learner_2
    CNTK::Learner.training_parameter_schedule(1, :minibatch)
    CNTK::Learner.training_parameter_schedule([1,2,3], :minibatch)
    CNTK::Learner.training_parameter_schedule([1,2,3], :minibatch, 100)

    CNTK::Learner.momentum_schedule(1)
  end

  def test_learner_3
    AdditionalLearningOptions.new
  end
end
