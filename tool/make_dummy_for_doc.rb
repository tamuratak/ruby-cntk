require "cntk"

include CNTK

# CNTK.constants.sort.delete_if{|klass| CNTK.const_get(klass).class == Class }

p CNTK::DictionaryValue.constants
exit 
[Axis, Constant, DeviceDescriptor, Dictionary, DictionaryValue, DistributedCommunicator, DistributedLearner, DistributedWorkerDescriptor,
 Function, Learner, 
 MinibatchData, MinibatchInfo, MinibatchSource, MinibatchTable, MomentumAsTimeConstantSchedule, MomentumSchedule,
 NDArrayView, NDShape, Ops, Parameter, StreamInformation, Trainer,
 TrainingParameterPerMinibatchSchedule, TrainingParameterPerSampleSchedule,
 TrainingSession, Value, Variable].uniq.each{|klass|
  case klass
  when klass == Parameter || klass == Constant
    methods = klass.instance_methods(false) - Variable.instance_methods(false)
  else
    methods = klass.instance_methods(false)
  end
  puts "class " + klass.name
  smethods = klass.methods(false)
  smethods.sort.each{|m|
    puts "  def #{klass.name}::" + m.to_s
    puts "  end"
    puts
  }
  methods.sort.each{|m|
    puts "  def " + m.to_s
    puts "  end"
    puts
  }
  puts "end"
  puts 
}
