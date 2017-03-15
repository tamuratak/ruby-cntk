module CNTK

# constants for data type
DataType_Double
DataType_Float
DataType_Unknown

# constants for device
DeviceKind_CPU
DeviceKind_GPU

# constants for the storage format of NDArrayView
StorageFormat_Dense
StorageFormat_SparseCSC
StorageFormat_SparseBlockCol

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

  # @param  n [Integer, Axis]
  # @return [Axis]
  def Axis::from_num(n)
  end

  # @return [Array<Axis>]
  def Axis::unknown_dynamic_axes
  end

  # @overload initialize(n)
  #   @param n [Integer]
  #
  # @overload initialize(name, ordered)
  #   @param name    [String]
  #   @param ordered [Boolean]  (true)
  def initialize
  end

  # @param other [Axis]
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

class BackPropState

  # @return [DeviceDescriptor]
  def device
  end

  def function
  end

  # @return [Hash<Variable, Value>]
  def saved_forward_prop_values
  end

end
class Constant < Variable

  # @param arg [Numo::SFloat, Numo::DFloat, Numeric, Array<Numeric>]
  # @return [Constant]
  def Constant::create(arg)
  end

  # @param n [Numeric]
  # @return [Constant]
  def Constant::scalar(n)
  end

  # @param other [Variable]
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

  # @param device [DeviceDescriptor]
  # @return [nil]
  def DeviceDescriptor::set_default_device(device)
  end

  # @return [DeviceDescriptor]
  def DeviceDescriptor::use_default_device
  end

  # @param other [DeviceDescriptor]
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
  
  # @param h [Hash]
  # @return [Dictionary]
  def Dictionary::create(h)
  end

  # @param filename [String]
  # @return [Dictionary] 
  def Dictionary::load(filename)
  end

  def ==(other)
  end

  # @param key [String]
  # @return [DictionaryValue]
  def [](key)
  end

  # @param key [String]
  # @param val [DictionaryValue]
  def []=(key, val)
  end

  # @param  [Dictionary] other
  # @return [nil]
  def add(other)
  end

  # @param key [String]
  # @return [Boolean]
  def contains(key)
  end

  # @param filename [String]
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

  # @param h [Hash, Boolean, Numeric, String, NDShape, Axis, Array<DictionaryValue>, Dictionary, NDArrayView]
  # @return [DictionaryValue]
  def DictionaryValue::create(h)
  end

  # @param filename [String]
  # @return [DictionaryValue]
  def DictionaryValue::load(filename)
  end

  # @return [String]
  def DictionaryValue::type_name
  end

  # @param other [DictionaryValue]
  # @return [Boolean]
  def ==(other)
  end

  # @return [Boolean]
  def has_value
  end

  # @param filename [String]
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

  # @param dict   [Dictionary]
  # @param device [DeviceDescriptor]
  # @return [Function]
  def Function::deserialize(dict, device)
  end

  # @param filename [String]
  # @return [Function]
  def Function::load_model(filename)
  end

  # @param other [Variable, Function]
  # @return [Function]
  def <<(other)
  end

  # @param other [Function]
  # @return [Function]
  def >>(other)
  end

  # @return [Array<Variable>]
  def arguments
  end

  # @return [Dictionary]
  def attributes
  end

  # @param state          [BackPropState]
  # @param root_gradients [Hash<Variable,Value>]
  # @param variables      [Array<Variable>]
  # @return [Hash<Variables,Value>] 
  def backward(state, root_gradients, variables)
  end

  def block_arguments_mapping
  end

  # @return [Function]
  def block_root
  end

  # @return [Function]
  def call(other)
  end

  # @param clone_type  [ParameterCloningMethod_Share, ParameterCloningMethod_Clone, ParameterCloningMethod_Freeze]
  # @param replacement [Hash<Variable,Variable>]
  # @return [Function]
  def clone(clone_type, replacement)
  end

  # @return [Array<Constant>]
  def constants
  end

  # @return [Integer]
  def current_version
  end

  # @param argsmap [Hash<Variable,...>]
  # @param opt     [Hash]
  # @option opt [DeviceDescriptor] :device
  # @return [Function]
  def eval(argsmap, opt = {})
  end

  def evaluate
  end

  # @param argsmap [Hash<Variable,...>]
  # @param opt     [Hash]
  # @option opt [Boolean] :keep_for_backward
  # @option opt [DeviceDescriptor] :device
  # @return [Function]
  def forward(argsmap, opt = {})
  end

  # @return [Array<Variable>]
  def inputs
  end

  # @return [Boolean]
  def is_block
  end

  # @return [Boolean]
  def is_composite
  end

  # @return [Boolean]
  def is_primitive
  end

  # @return [String]
  def name
  end

  # @return [String]
  def op_name
  end

  # @return [Variable]
  def output
  end

  # @return [Array<Variable>]
  def outputs
  end

  # @return [Array<Parameter>]
  def parameters
  end

  # @return [Array<Variable>]
  def placeholders
  end

  # @return [nil]
  def print_graph
  end

  # @param var [Variable]
  def replace_placeholder(var)
  end

  # @param h [Hash<Variable, Variable>]
  def replace_placeholders(h)
  end

  # @param filename [String]
  # @return [nil]
  def restore_model(filename)
  end

  # @return [Function]
  def root_function
  end

  # @param filename [String]
  # @return [nil]
  def save_model(filename)
  end

  # @return [Dictionary]
  def serialize
  end

  # @param name [String]
  # @return [nil]
  def set_name(name)
  end

  # @return [String]
  def uid
  end

end

class Learner

  # @return [Dictionary]
  def create_checkpoint
  end

  # @return [Float]
  def learning_rate
  end

  # @return [Array<Parameter>]
  def parameters
  end

  # @param schedule [LearningRateSchedule]
  # @return [nil]
  def reset_learning_rate(schedule)
  end

  # @return [nil]
  def reset_smoothed_gradients
  end

  # @param dict [Dictionary]
  def restore_from_checkpoint(dict)
  end

  # @return [Integer]
  def total_number_of_samples_seen
  end

  # @param grad      [Hash<Parameter,NDArrayView>]
  # @param count     [Integer]
  # @param sweep_end [Boolean]
  # @return [Boolean]
  def update(grad, count, sweep_end=false)
  end

end

class MinibatchData

  # @!attribute [rw] data
  #   @return [Value]
  attr :data

  # @!attribute [rw] number_of_samples
  #  @return [Integer]
  attr :number_of_samples

  # @!attribute [rw] number_of_sequences
  #  @return [Integer]
  attr :number_of_sequences

  # @!attribute [rw] sweep_end
  #   @return [Boolean]
  attr :sweep_end

end

class MinibatchInfo

  # @!attribute [rw] at_end_of_data
  #   @return [Boolean]
  attr :at_end_of_data

  # @!attribute [rw] eval_criterion_value
  #   @return [NDArrayView]

  # @return [Boolean]
  def is_empty
  end

  # @!attribute [rw] number_of_samples
  #   @return [Integer]
  attr :number_of_samples

  # @!attribute [rw] training_loss_value
  #   @return [NDArrayView]
  attr :training_loss_value

end

class MinibatchSource

  # @return [Dictionary]
  def get_checkpoint_state
  end

  # @return [Hash<StreamInformation, MinibatchData>]
  def get_next_minibatch
  end

  # @param dict [Dictionary]
  # @return [nil]
  def restore_from_checkpoint(dict)
  end

  # @param name [String]
  # @return [StreamInformation]
  def stream_info(name)
  end

  # @return [Array<StreamInformation>]
  def stream_infos
  end

end

class MinibatchTable

  def []
  end

  def []=
  end

end

class MomentumAsTimeConstantSchedule

  # @param val [Float]
  # @return [MomentumAsTimeConstantSchedule]
  def initialize(val)
  end

end

class MomentumSchedule

  # @param val [Float]
  # @return [MomentumSchedule]
  def initialize(val)
  end

end

class NDArrayView

  # @param arry [Numo::NArray]
  # @return [NDArrayView]
  def NDArrayView::create(arry)
  end

  def alias
  end

  # @param device [DeviceDescriptor]
  # @return [nil]
  def change_device(device)
  end

  # @param ndarray [NDArrayView]
  # @return [nil]
  def copy_from(ndarray)
  end

  # @param device    [DeviceDescriptor]
  # @param read_only [Boolean]
  # @return [NDArrayView]
  def deep_clone(device, read_only = false)
  end

  # @return [DataType_Float, DataType_Double, DataType_Unknown]
  def get_data_type
  end

  # @return [StorageFormat_Dense, StorageFormat_SparseCSC, StorageFormat_SparseBlockCol]
  def get_storage_format
  end

  # @return [Boolean]
  def is_read_only
  end

  # @return [Boolean]
  def is_sparse
  end

  # @param val [Float]
  # @return [nil]
  def set_value(val)
  end

  # @return [NDShape]
  def shape
  end

  # @return [Numo::NArray]
  def to_narray
  end

  # @return [Array<Float>]
  def to_vec
  end

end

class NDShape

  # @return [NDShape]
  def NDShape::unknown
  end

  # @param other [NDShape]
  # @return [Boolean]
  def ==(other)
  end

  # @param shape [NDShape]
  # @return [NDShape]
  def append_shape(shape)
  end

  # @return [Array<Integer>]
  def dimensions
  end

  # @return [Boolean]
  def is_unknown
  end

  # @return [Integer]
  def rank
  end

  # @return [Array<Integer>]
  def reverse
  end

  # @return [NDShape]
  def sub_shape(begin_index, end_index)
  end

  # @return [Array<Integer>]
  def to_a
  end

  # @return [Array<Integer>]
  def to_ary
  end

  # @return [Integer]
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

  # @return [Integer]
  def current_value_time_stamp
  end

  # @return [nil]
  def record_value_update
  end

  # @param ndarray [NDArrayView]
  # @return [nil]
  def set_value(ndarray)
  end

  # @return [NDArrayView]
  def value
  end

end

class StreamInformation

  # @param other [StreamInformation]
  # @return [Boolean]
  def ==(other)
  end

  # @!attribute [rw] element_type
  #   @return [DataType_Double, DataType_Float, DataType_Unknown]
  attr :element_type

  # @!attribute [rw] id
  #   @return [Integer]
  attr :id

  # @!attribute [rw] name
  #   @return [String]

  # @!attribute [rw] sample_layout
  #   @return [NDShape]
  attr :sample_layout

  # @!attribute [rw] storage_format
  #   @return [StorageFormat_Dense, StorageFormat_SparseCSC, StorageFormat_SparseBlockCol]
  attr :storage_format

end

class Trainer

  def Trainer::create
  end

  # @return [Function]
  def evaluation_function
  end

  # @return [Function]
  def loss_function
  end

  # @return [Function]
  def model
  end

  # @return [Array<Learner>]
  def parameter_learners
  end

  # @return [Float]
  def previous_minibatch_evaluation_average
  end

  # @return [Float]
  def previous_minibatch_loss_average
  end

  # @return [Integer]
  def previous_minibatch_sample_count
  end

  # @param filename [String]
  # @return [nil]
  def restore_from_checkpoint(filename)
  end

  # @param filename [String]
  # @return [nil]
  def save_checkpoint(filename)
  end

  def test_minibatch
  end

  # @return [Integer]
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

  # @param val [NDArrayView, Numo::NArray]
  # @return [Value]
  def Value::create(val)
  end

  # @param val [NDArrayView]
  # @return [Value]
  def initialize(val)
  end

  # @param read_only [Boolean]
  # @return [Value]
  def alias(read_only=false)
  end

  # @param val [Value]
  # @return [nil]
  def copy_from(val)
  end

  def copy_variable_value_to
  end

  # @return [NDArrayView]
  def data
  end

  # @param read_only [Boolean]
  # @return [Value]
  def deep_clone(read_only=true)
  end

  # @return [DeviceDescriptor]
  def device
  end

  # @return [DataType_Float, DataType_Double, DataType_Unknown]
  def get_data_type
  end

  # @return [StorageFormat_Dense, StorageFormat_SparseCSC, StorageFormat_SparseBlockCol]
  def get_storage_format
  end

  # @return [Boolean]
  def is_read_only
  end

  # @return [Boolean]
  def is_sparse
  end

  def mask
  end

  # @return [Integer]
  def masked_count
  end

  # @param shape [Array<Integer>]
  # @return [Value]
  def reshape(shape)
  end

  # @return [NDShape]
  def shape
  end

  # @return [Numo::NArray]
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
