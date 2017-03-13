module CNTK

# constants for data type
DataType_Double
DataType_Float
DataType_Unknown

# constants for device
DeviceKind_CPU
DeviceKind_GPU

class Axis

  # @return [Axis]
  def Axis::all_static_axes
  end

  # @return [Axis]
  def Axis::default_batch_axis
  end

  # @return [Axis]
  def Axis::default_dynamic_axis
  end

  # @return [Array<Axis>]
  def Axis::default_input_variable_dynamic_axes
  end

  # @return [Axis]
  def Axis::end_static_axis
  end

  # @param  [Integer, Axis] n
  # @return [Axis]
  def Axis::from_num(n)
  end

  # @return [Array<Axis>]
  def Axis::unknown_dynamic_axes
  end

  # @overload initialize(n)
  #   @param [Integer] n
  #
  # @overload initialize(name, ordered)
  #   @param [String]  name
  #   @param [Boolean] ordered (true)
  def initialize
  end

  # @param [Axis] other
  # @return [Boolean]
  def ==(other)
  end

  # @return [Boolean]
  def is_dynamic_axis
  end

  # @return [Boolean]
  def is_ordered
  end

  # @return [Boolean]
  def is_static_axis
  end

  # @return [String]
  def name
  end

  # @return [Integer]
  def static_axis_index
  end

end

class Constant < Variable

  # @param  [Numo::SFloat, Numo::DFloat, Numeric, Array<Numeric>] arg
  # @return [Constant]
  def Constant::create(arg)
  end

  # @param  [Numeric] n
  # @return [Constant]
  def Constant::scalar(n)
  end

  # @param  [Variable] other
  # @return [Function]
  def *(other)
  end

  # @return [Value]
  def value
  end

end

class DeviceDescriptor

  # @return [Array<DeviceDescriptor>]
  def DeviceDescriptor::all_devices
  end

  # @return [DeviceDescriptor]
  def DeviceDescriptor::best_device
  end

  # @return [DeviceDescriptor]
  def DeviceDescriptor::cpudevice
  end

  # @return [DeviceDescriptor]
  def DeviceDescriptor::default_device
  end

  # @return [DeviceDescriptor]
  def DeviceDescriptor::gpudevice
  end

  # @param [DeviceDescriptor] device
  # @return [nil]
  def DeviceDescriptor::set_default_device(device)
  end

  # @return [DeviceDescriptor]
  def DeviceDescriptor::use_default_device
  end

  # @param  [DeviceDescriptor] other
  # @return [Boolean]
  def ==(other)
  end

  # @return [Integer]
  def id
  end

  # @return [DeviceKind_CPU, DeviceKind_GPU]
  def type
  end

end

class Dictionary
  
  # @param  [Hash]
  # @return [Dictionary]
  def Dictionary::create(h)
  end

  # @param  [String] filename
  # @return [Dictionary] 
  def Dictionary::load(filename)
  end

  def ==(other)
  end

  # @param  [String] key
  # @return [DictionaryValue]
  def [](key)
  end

  # @param  [String] key
  # @param  [DictionaryValue] val
  def []=(key, val)
  end

  # @param  [Dictionary] other
  # @return [nil]
  def add(other)
  end

  # @param  [String]  key
  # @return [Boolean]
  def contains(key)
  end

  # @param [String] filename
  # @return [nil]
  def save(filename)
  end

  # @return [Integer]
  def size
  end

end

class DictionaryValue

  # constants for types
  Type_None
  Type_Bool
  Type_Int
  Type_SizeT
  Type_Float
  Type_Double
  Type_String
  Type_NDShape
  Type_Axis
  Type_Vector
  Type_Dictionary
  Type_NDArrayView

  # @param  [Hash, Boolean, Numeric, String, NDShape, Axis, Array<DictionaryValue>, Dictionary, NDArrayView]
  # @return [DictionaryValue]
  def DictionaryValue::create(h)
  end

  # @param  [String] filename
  # @return [DictionaryValue]
  def DictionaryValue::load(filename)
  end

  # @return [String]
  def DictionaryValue::type_name
  end

  # @param  [DictionaryValue] other
  # @return [Boolean]
  def ==(other)
  end

  # @return [Boolean]
  def has_value
  end

  # @param [String] filename
  def save(filename)
  end

  # @return [Boolean, Numeric, String, NDShape, Axis, Array<DictionaryValue>, Dictionary, NDArrayView]
  def value
  end

  # @return [Type_None, Type_Bool, Type_Int, Type_SizeT, Type_Float, Type_Double, Type_String, Type_NDShape, Type_Axis, Type_Vector, Type_Dictionary, Type_NDArrayView]
  def value_type
  end

end

class DistributedCommunicator

  def DistributedCommunicator::finalize
  end

  def aggregate
  end

  def aggregate_in_place
  end

  def barrier
  end

  def concatenate
  end

  def current_worker
  end

  def gather
  end

  def sub_group
  end

  def workers
  end

end

class DistributedLearner

  def get_communicator
  end

  def learning_rate
  end

  def parallelization_after
  end

  def reset_learning_rate
  end

  def reset_smoothed_gradients
  end

  def update
  end

end

class DistributedWorkerDescriptor

  def ==
  end

  def is_main
  end

  def m_global_rank
  end

  def m_global_rank=
  end

  def m_host_id
  end

  def m_host_id=
  end

end

class Function

  def Function::deserialize
  end

  def Function::load_model
  end

  # @param  [Variable, Function] other
  # @return [Function]
  def <<(other)
  end

  # @param  [Function] other
  # @return [Function]
  def >>(other)
  end

  # @return [Array<Variable>]
  def arguments
  end

  def attributes
  end

  def backward
  end

  def block_arguments_mapping
  end

  def block_root
  end

  def call
  end

  def clone
  end

  def coerce
  end

  def constants
  end

  def convert_to_value
  end

  def current_version
  end

  def eval
  end

  def evaluate
  end

  def forward
  end

  def inputs
  end

  def is_block
  end

  def is_composite
  end

  def is_primitive
  end

  def name
  end

  def op_name
  end

  # @return [Variable]
  def output
  end

  # @return [Array<Variable>]
  def outputs
  end

  def parameters
  end

  def placeholders
  end

  def print_graph
  end

  def remove_dynamic_axes
  end

  def replace_placeholder
  end

  def replace_placeholders
  end

  def restore_model
  end

  def root_function
  end

  def save_model
  end

  def serialize
  end

  def set_name
  end

  def uid
  end

end

class Learner
  def Learner::adagrad
  end

  def Learner::adam_sgd
  end

  def Learner::momentum_as_time_constant_schedule
  end

  def Learner::momentum_schedule
  end

  def Learner::momentum_sgd
  end

  def Learner::nesterov
  end

  def Learner::rmsprop
  end

  def Learner::sgd
  end

  def Learner::training_parameter_schedule
  end

  def create_checkpoint
  end

  def learning_rate
  end

  def parameters
  end

  def reset_learning_rate
  end

  def reset_smoothed_gradients
  end

  def restore_from_checkpoint
  end

  def total_number_of_samples_seen
  end

  def update
  end

end

class MinibatchData
  def data
  end

  def data=
  end

  def inspect
  end

  def number_of_samples
  end

  def number_of_samples=
  end

  def number_of_sequences
  end

  def number_of_sequences=
  end

  def sweep_end
  end

  def sweep_end=
  end

end

class MinibatchInfo
  def at_end_of_data
  end

  def at_end_of_data=
  end

  def eval_criterion_value
  end

  def eval_criterion_value=
  end

  def is_empty
  end

  def number_of_samples
  end

  def number_of_samples=
  end

  def training_loss_value
  end

  def training_loss_value=
  end

end

class MinibatchSource
  def get_checkpoint_state
  end

  def get_next_minibatch
  end

  def restore_from_checkpoint
  end

  def stream_info
  end

  def stream_infos
  end

end

class MinibatchTable
  def []
  end

  def []=
  end

  def __get__
  end

  def begin
  end

  def clear
  end

  def count
  end

  def delete
  end

  def dup
  end

  def each
  end

  def each_key
  end

  def each_value
  end

  def empty?
  end

  def end
  end

  def entries
  end

  def erase
  end

  def find
  end

  def get_allocator
  end

  def has_key?
  end

  def include?
  end

  def inspect
  end

  def key_iterator
  end

  def keys
  end

  def select
  end

  def size
  end

  def swap
  end

  def to_a
  end

  def to_s
  end

  def value_iterator
  end

  def values
  end

  def values_at
  end

end

class MomentumAsTimeConstantSchedule
end

class MomentumSchedule
end

class NDArrayView
  def NDArrayView::create
  end

  def alias
  end

  def change_device
  end

  def copy_from
  end

  def deep_clone
  end

  def get_data_type
  end

  def get_storage_format
  end

  def is_read_only
  end

  def is_sparse
  end

  def set_value
  end

  def shape
  end

  def to_narray
  end

  def to_vec
  end

end

class NDShape
  def NDShape::unknown
  end

  def ==
  end

  def append_shape
  end

  def dimensions
  end

  def inspect
  end

  def is_unknown
  end

  def rank
  end

  def reverse
  end

  def sub_shape
  end

  def to_a
  end

  def to_ary
  end

  def total_size
  end

end

class Ops
  def Ops::abs
  end

  def Ops::alias
  end

  def Ops::batch_normalization
  end

  def Ops::binary_cross_entropy
  end

  def Ops::ceil
  end

  def Ops::classification_error
  end

  def Ops::clip
  end

  def Ops::combine
  end

  def Ops::constant
  end

  def Ops::convert_to_one_variable
  end

  def Ops::convert_to_pooling_type
  end

  def Ops::convert_to_variable
  end

  def Ops::convolution
  end

  def Ops::cos
  end

  def Ops::cosine_distance
  end

  def Ops::cross_entropy_with_softmax
  end

  def Ops::dropout
  end

  def Ops::edit_distance_error
  end

  def Ops::element_divide
  end

  def Ops::element_select
  end

  def Ops::element_times
  end

  def Ops::equal
  end

  def Ops::exp
  end

  def Ops::floor
  end

  def Ops::future_value
  end

  def Ops::greater
  end

  def Ops::greater_equal
  end

  def Ops::hardmax
  end

  def Ops::highest_precision_type
  end

  def Ops::input_variable
  end

  def Ops::lambda_rank
  end

  def Ops::less
  end

  def Ops::less_equal
  end

  def Ops::log
  end

  def Ops::log_add_exp
  end

  def Ops::minus
  end

  def Ops::ndcg_at_1
  end

  def Ops::negate
  end

  def Ops::not_equal
  end

  def Ops::output_variable
  end

  def Ops::parameter
  end

  def Ops::past_value
  end

  def Ops::placeholder_variable
  end

  def Ops::plus
  end

  def Ops::pooling
  end

  def Ops::random_sample
  end

  def Ops::random_sample_inclusion_frequency
  end

  def Ops::reciprocal
  end

  def Ops::reduce_log_sum_exp
  end

  def Ops::reduce_max
  end

  def Ops::reduce_mean
  end

  def Ops::reduce_min
  end

  def Ops::reduce_prod
  end

  def Ops::reduce_sum
  end

  def Ops::relu
  end

  def Ops::reshape
  end

  def Ops::reverse_dynamic_axes
  end

  def Ops::reverse_reshape_axis
  end

  def Ops::roipooling
  end

  def Ops::round
  end

  def Ops::sigmoid
  end

  def Ops::sin
  end

  def Ops::slice
  end

  def Ops::softmax
  end

  def Ops::splice
  end

  def Ops::sqrt
  end

  def Ops::square
  end

  def Ops::squared_error
  end

  def Ops::tanh
  end

  def Ops::times
  end

  def Ops::times_transpose
  end

  def Ops::transpose
  end

  def Ops::unpooling
  end

  def Ops::weighted_binary_cross_entropy
  end

end

class Parameter < Variable
  def Parameter::create
  end

  def current_value_time_stamp
  end

  def record_value_update
  end

  def set_value
  end

  def value
  end

end

class StreamInformation
  def ==
  end

  def element_type
  end

  def element_type=
  end

  def id
  end

  def id=
  end

  def inspect
  end

  def name
  end

  def name=
  end

  def sample_layout
  end

  def sample_layout=
  end

  def storage_format
  end

  def storage_format=
  end

end

class Trainer
  def Trainer::create
  end

  def evaluation_function
  end

  def loss_function
  end

  def model
  end

  def parameter_learners
  end

  def previous_minibatch_evaluation_average
  end

  def previous_minibatch_loss_average
  end

  def previous_minibatch_sample_count
  end

  def restore_from_checkpoint
  end

  def save_checkpoint
  end

  def test_minibatch
  end

  def total_number_of_samples_seen
  end

  def train_minibatch
  end

end

class TrainingParameterPerMinibatchSchedule
  def []
  end

end

class TrainingParameterPerSampleSchedule
  def []
  end

end

class TrainingSession
  def get_minibatch_size
  end

  def on_checkpoint_end
  end

  def on_checkpoint_start
  end

  def on_cross_validation_end
  end

  def on_cross_validation_start
  end

  def on_minibatch_end
  end

  def on_minibatch_start
  end

  def on_progress
  end

  def restore_from_checkpoint
  end

  def train
  end

end

class Value
  def Value::create
  end

  def alias
  end

  def copy_from
  end

  def copy_variable_value_to
  end

  def data
  end

  def deep_clone
  end

  def get_data_type
  end

  def get_storage_format
  end

  def inspect
  end

  def is_read_only
  end

  def is_sparse
  end

  def mask
  end

  def masked_count
  end

  def reshape
  end

  def shape
  end

  def to_narray
  end

end

class Variable
  def *
  end

  def +
  end

  def -
  end

  def -@
  end

  def ==
  end

  def __dynamic_axes__
  end

  def coerce
  end

  def dynamic_axes
  end

  def get_data_type
  end

  def is_constant
  end

  def is_input
  end

  def is_output
  end

  def is_parameter
  end

  def is_placeholder
  end

  def is_scalar
  end

  def is_sparse
  end

  def kind
  end

  def name
  end

  def needs_gradient
  end

  def owner
  end

  def shape
  end

  def to_function
  end

  def uid
  end

end
end # module CNTK
