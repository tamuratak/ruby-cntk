require "cntk"
require "numo/narray"

module MNISTExample
class << self  
  include CNTK
  include CNTK::Layers
  include CNTK::Ops

  def reader(filename)
    dict = { 
      epochSize: CNTK::MinibatchSource::InfinitelyRepeat,
      verbosity: 0,
      randomize: true,
      randomizationWindow: CNTK::MinibatchSource::DefaultRandomizationWindow,
      deserializers: 
      [{ 
         type: "CNTKTextFormatDeserializer",
         file: File.join(File.dirname(__FILE__), filename),
         maxErrors: 100,
         skipSequenceIds: false,
         traceLevel: 0,
         input: {
           x: { dim: 784, format: "dense"},
           y: { dim:  10, format: "dense"},
         }
       }]
    }
    CNTK.create_composite_minibatch_source(dict)
  end

  def model
    x = input_variable([784])
    f = dense([400]) >> relu() >> dense([400]) >> relu() >> dense([10]) 
    [f.(x/256.0), x]
  end

  def trainer(model, loss_function, evaluate_function)
    sch = CNTK::Learner.training_parameter_schedule(0.5, :minibatch)
    learner = CNTK::Learner.sgd(model.parameters, sch)
    CNTK::Trainer::create(model: model, loss: loss_function, evaluation: evaluate_function, learners: [learner])
  end

end
end

mnist             = MNISTExample
train_reader      = mnist.reader("mnist_train.txt")
test_reader       = mnist.reader("mnist_test.txt")
model, x          = mnist.model
y                 = CNTK::Ops.input_variable(model.output.shape)
loss_function     = CNTK::Ops.cross_entropy_with_softmax(model.output, y)
evaluate_function = CNTK::Ops.classification_error(model.output, y)
trainer           = mnist.trainer(model, loss_function, evaluate_function)

minibatch_size           = 64
num_samples_per_sweep    = 60000
num_sweeps_to_train_with = 10
num_minibatches_to_train = (num_samples_per_sweep * num_sweeps_to_train_with) / minibatch_size

num_minibatches_to_train.times {|i|
  minibatch = train_reader.next_minibatch(minibatch_size)
  trainer.train_minibatch({x => minibatch["x"], y => minibatch["y"]})
  if i % 500 == 0
    puts( "Minibatch: %4d, Loss: %.4f, Error: %5.2f%%" %
          [i, 
           trainer.previous_minibatch_loss_average,
           trainer.previous_minibatch_evaluation_average * 100] )           
  end
}

test_minibatch_size     = 512
num_samples             = 10000
num_minibatches_to_test = num_samples / test_minibatch_size
test_result             = 0.0

num_minibatches_to_test.times{
  minibatch   = test_reader.next_minibatch(test_minibatch_size)
  eval_error  = trainer.test_minibatch({ x => minibatch["x"], y => minibatch["y"] })
  test_result = test_result + eval_error
}

puts("Average test error: %3.3f%%" % (test_result * 100 / num_minibatches_to_test))
