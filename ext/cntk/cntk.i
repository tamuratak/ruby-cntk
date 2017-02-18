%module CNTK
#pragma SWIG nowarn=801

 // 
 // The file is based on the followings
 // https://github.com/Microsoft/CNTK/blob/master/bindings/python/cntk/cntk_py.i
 // https://github.com/Microsoft/CNTK/blob/master/Source/CNTKv2LibraryDll/API/CNTKLibrary.h
 //

%include <stl.i>
%include <std_wstring.i>
%include <std_vector.i>
%include <std_map.i>
%include <std_unordered_map.i>
%include <std_unordered_set.i>
%include <std_pair.i>
%include <std_shared_ptr.i>

%template(StdVectorSizeT) std::vector<size_t>;
%template(StdVectorDouble) std::vector<double>;
%template(StdVectorFloat) std::vector<float>;
%template(StdVectorVectorSizeT) std::vector<std::vector<size_t> >;
%template(StdVectorVectorFloat) std::vector<std::vector<float> >;
%template(StdVectorVectorDouble) std::vector<std::vector<double> >;

%shared_ptr(RubyCNTK::Trainer)
%shared_ptr(RubyCNTK::TrainingSession)
%shared_ptr(RubyCNTK::BasicTrainingSession)
%shared_ptr(RubyCNTK::Function)
%shared_ptr(RubyCNTK::NDArrayView)
%shared_ptr(RubyCNTK::Value)
%shared_ptr(RubyCNTK::NDMask)
%shared_ptr(RubyCNTK::BackPropState)
%shared_ptr(RubyCNTK::Learner)
%shared_ptr(RubyCNTK::MinibatchSource)
%shared_ptr(RubyCNTK::DistributedCommunicator)
%shared_ptr(RubyCNTK::QuantizedDistributedCommunicator)
%shared_ptr(RubyCNTK::DistributedLearner)


%{
#include <CNTKLibrary.h>
%}
%inline %{
namespace CNTK {};
namespace RubyCNTK {
  using namespace CNTK;
  static RubyCNTK::DeviceDescriptor __cpu_device__ = RubyCNTK::DeviceDescriptor::CPUDevice();
  static RubyCNTK::DeviceDescriptor __best_device__ = RubyCNTK::DeviceDescriptor::BestDevice();
  static std::vector<RubyCNTK::DeviceDescriptor> __all_device__;
};
%}

namespace RubyCNTK {
  typedef std::shared_ptr<RubyCNTK::NDArrayView> NDArrayViewPtr;
  typedef std::shared_ptr<RubyCNTK::Dictionary> DictionaryPtr;
  typedef std::shared_ptr<RubyCNTK::Function> FunctionPtr;
  typedef std::shared_ptr<RubyCNTK::Value> ValuePtr;
  typedef std::shared_ptr<RubyCNTK::NDMask> NDMaskPtr;
  typedef std::shared_ptr<RubyCNTK::BackPropState> BackPropStatePtr;
  typedef std::shared_ptr<RubyCNTK::Learner> LearnerPtr;
  typedef std::shared_ptr<RubyCNTK::DistributedCommunicator> DistributedCommunicatorPtr;
  typedef std::shared_ptr<RubyCNTK::QuantizedDistributedCommunicator> QuantizedDistributedCommunicatorPtr;
  typedef std::shared_ptr<RubyCNTK::Trainer> TrainerPtr;
  typedef std::shared_ptr<RubyCNTK::MinibatchSource> MinibatchSourcePtr;
  typedef std::shared_ptr<RubyCNTK::TrainingSession> TrainingSessionPtr;
};

%template() std::vector<RubyCNTK::DeviceDescriptor>;
%template(StdVectorVariable) std::vector<RubyCNTK::Variable>;
%template(StdVectorStdPairVarableVariable) std::vector<std::pair<RubyCNTK::Variable, RubyCNTK::Variable> >;
%template(StdUMapVariableValue) std::unordered_map< RubyCNTK::Variable, RubyCNTK::ValuePtr >;
%template(StdUMapVariableVariable) std::unordered_map< RubyCNTK::Variable, RubyCNTK::Variable >;
%template(StdUSetVariable) std::unordered_set<RubyCNTK::Variable>;
%template(StdUSetDistributedWorkerDescriptor) std::unordered_set<RubyCNTK::DistributedWorkerDescriptor>;


///************************************
/// renaming rule
///
///************************************
%rename("__%(utitle)s__", %$isfunction, notregexmatch$name="Initializer$") "";
%rename("%(utitle)s", %$isfunction, regexmatch$name="Initializer$") "";
%rename("%(utitle)s", %$ismember, %$isfunction) "";
%rename("%(utitle)s", %$ismember, %$isvariable) "";
%rename("%s", %$isenum) "";
%rename("%s", %$isconstructor) "";
%rename(__forward__) RubyCNTK::Function::Forward;

%typecheck(1000) RubyCNTK::NDShape const &, RubyNTK::NDShape {
    // '1000' is the typecheck precedence code. It means: check after basic
    // types, but before arrays. See: http://www.swig.org/Doc3.0/Typemaps.html#Typemaps_overloading
  $1 = NIL_P(rb_check_array_type($input)) ? 0 : 1;
}

%typemap(in) const RubyCNTK::NDShape& (RubyCNTK::NDShape tmp) {
  VALUE arry = rb_check_array_type($input);
  if(NIL_P(arry)) {
    rb_raise(rb_eArgError, "Array expected"); SWIG_fail;
  }else{
    std::vector<size_t> dimensions(RARRAY_LEN(arry));
    for (int i=0; i<RARRAY_LEN(arry); i++) {
	VALUE elt = RARRAY_AREF(arry, i);
        dimensions[i] = NUM2INT(elt);
    }
    tmp = CNTK::NDShape(dimensions);
    $1 = &tmp;
  }
}

//
// Exception handling
//
%exception {
    try { $action }
    catch (const std::runtime_error &e) { SWIG_exception(SWIG_RuntimeError,e.what()); }
    catch (const std::invalid_argument &e) { SWIG_exception(SWIG_ValueError,e.what()); }
    catch (const std::logic_error &e) { SWIG_exception(SWIG_RuntimeError,e.what()); }
    catch (...) { SWIG_exception(SWIG_UnknownError,"Runtime exception"); }
}

//
// In the future, we would just swtich to %include "CNTKLibrary.h". 
//  
namespace RubyCNTK {

  enum class DataType {
    Unknown,
    Float,
    Double
  };

  const char* DataTypeName(enum DataType);
  size_t DataTypeSize(enum DataType);

  enum class StorageFormat
  {
    Dense,
    SparseCSC,
    SparseBlockCol,
  };

  enum class DeviceKind
  {
    CPU,
    GPU,
  };

  struct MinibatchInfo
  {
    bool atEndOfData;
    //    bool atEndOfSweep;
    size_t numberOfSamples;
    NDArrayViewPtr trainingLossValue;
    NDArrayViewPtr evalCriterionValue;
    
    bool IsEmpty();
  };

%nodefaultctor DeviceDescriptor;
  class DeviceDescriptor 
  {
  public:
    unsigned int Id();
    RubyCNTK::DeviceKind Type();

    %extend{

      %newobject CPUDevice;
      static RubyCNTK::DeviceDescriptor* CPUDevice(){
        return new RubyCNTK::DeviceDescriptor(RubyCNTK::DeviceDescriptor::CPUDevice());
      }

      %newobject GPUDevice;
      static RubyCNTK::DeviceDescriptor* GPUDevice(unsigned int deviceId){
        return new RubyCNTK::DeviceDescriptor(RubyCNTK::DeviceDescriptor::GPUDevice(deviceId));
      }

      %newobject DefaultDevice;
      static RubyCNTK::DeviceDescriptor* DefaultDevice(){
        return new RubyCNTK::DeviceDescriptor(RubyCNTK::DeviceDescriptor::DefaultDevice());
      }

      %newobject UseDefaultDevice;
      static RubyCNTK::DeviceDescriptor* UseDefaultDevice(){
        return new RubyCNTK::DeviceDescriptor(RubyCNTK::DeviceDescriptor::UseDefaultDevice());
      }

      %newobject BestDevice;
      static RubyCNTK::DeviceDescriptor* BestDevice(){
        return new RubyCNTK::DeviceDescriptor(RubyCNTK::DeviceDescriptor::BestDevice());
      }

      bool __eq__(const DeviceDescriptor& other){
        return (*$self) == other;
      }

    }

    static void SetDefaultDevice(const DeviceDescriptor& newDefaultDevice);
    static const std::vector<DeviceDescriptor>& AllDevices();

  };

%nodefaultctor DeviceDescriptor;

  class NDShape {
  public:
    NDShape();
    NDShape(size_t);
    NDShape(size_t, size_t);
    NDShape(const std::vector<size_t>&);
    ~NDShape();

    const std::vector<size_t>& Dimensions();

    bool IsUnknown();

    size_t Rank();

    NDShape SubShape();
    NDShape SubShape(size_t);
    NDShape SubShape(size_t, size_t);
    size_t TotalSize();

    NDShape AppendShape(const NDShape&);

    %extend {
      size_t __getitem__(size_t axis) {
        return (*$self)[axis];
      }

      void __setitem__(size_t axis, size_t n) {
        (*$self)[axis] = n;
      }

      bool __eq__(const NDShape& other) {
        return (*$self) == other;
      }

    }
  };

  class NDArrayView
  {
  public:
    NDArrayView(enum DataType, enum StorageFormat, const NDShape&, const DeviceDescriptor&);    
    NDArrayView(double);
    ~NDArrayView();

    //    DeviceDescriptor Device();
    enum DataType GetDataType();
    enum StorageFormat GetStorageFormat();
    const NDShape& Shape();
    bool IsSparse();
    bool IsReadOnly();

    void SetValue(float value);
    void SetValue(double value);

    NDArrayViewPtr DeepClone(const DeviceDescriptor& device, bool readOnly = false);
    
    NDArrayViewPtr DeepClone(bool readOnly);
    NDArrayViewPtr DeepClone();
    NDArrayViewPtr Alias(bool readOnly = false);
    void CopyFrom(const NDArrayView& source);
    void ChangeDevice(const DeviceDescriptor& device);

    %extend{

      NDArrayView(enum DataType dataType, const NDShape& shape, std::vector<double> v, const DeviceDescriptor& device, bool readOnly) {
        using namespace CNTK;
        if (dataType == DataType::Double) {
          NDArrayView tmp(shape, v.data(), v.size(), DeviceDescriptor::CPUDevice(), readOnly);
          auto x = new NDArrayView(DataType::Double, tmp.Shape(), device);
          x->CopyFrom(tmp);
          return x;
        }else if (dataType == DataType::Float) {
          std::vector<float> vf(v.size());
          vf.assign(v.begin(), v.end());
          NDArrayView tmp(shape, vf.data(), vf.size(), DeviceDescriptor::CPUDevice(), readOnly);
          auto x = new NDArrayView(DataType::Float, tmp.Shape(), device);
          x->CopyFrom(tmp);
          return x;
        }else{
          SWIG_exception(SWIG_RuntimeError, "double or float");
        };

      }

      std::vector<double> to_vec() {
        using namespace CNTK;
        size_t total = (*$self).Shape().TotalSize();
        std::vector<double> ret(total);
        DataType cntk_type = (*$self).GetDataType();
        NDArrayView* cpuView;
        void* buffer;

        if ((*self).Device() != DeviceDescriptor::CPUDevice()) {
          cpuView = new NDArrayView(cntk_type, (*$self).Shape(), DeviceDescriptor::CPUDevice());
          cpuView->CopyFrom((*$self));
        } else {
          cpuView = const_cast<NDArrayView*>(&(*$self));
        }

        if (cntk_type == DataType::Float) {
          std::vector<float> tmp(total);
          buffer = (void*)cpuView->DataBuffer<float>();
          memcpy(tmp.data(), buffer, sizeof(float) * total);
          ret.assign(tmp.begin(), tmp.end());
        } else if (cntk_type == DataType::Double) {
          buffer = (void*)cpuView->DataBuffer<double>();
          memcpy(ret.data(), buffer, sizeof(double) * total);
        } else {
          throw std::invalid_argument("unknown CNTK data type");
        }

        if ((*self).Device() != DeviceDescriptor::CPUDevice()) {
          delete cpuView;
        }

        return ret;
      }

    }
    
  };

  class Axis 
  {
  public:
    Axis(int);
    Axis(const std::wstring&);
    Axis(const std::wstring&, bool);
    ~Axis();

    bool IsStaticAxis();
    bool IsDynamicAxis();
    bool IsOrdered();
    int StaticAxisIndex(bool checked = true);
    const std::wstring& Name() ;

    static const std::vector<Axis>& DefaultInputVariableDynamicAxes();
    static const std::vector<Axis>& UnknownDynamicAxes();
    static const Axis& DefaultDynamicAxis();
    static const Axis& DefaultBatchAxis();
    static const Axis& AllStaticAxes();
    static Axis EndStaticAxis();
    
    %extend{
      bool __eq__(const Axis& other) {
        return (*$self) == other;
      }
    }
  };

  class DictionaryValue
  {
  public:
    enum class Type : unsigned int
    {
        None,
        Bool,
        Int,
        SizeT,
        Float,
        Double,
        String,
        NDShape,
        Axis,
        Vector,
        Dictionary,
        NDArrayView,
    };

    static const char* TypeName(Type type);
    DictionaryValue();
    ~DictionaryValue();

    DictionaryValue(bool);
    DictionaryValue(size_t);
    DictionaryValue(double);
    DictionaryValue(const std::vector<DictionaryValue>& value);
    DictionaryValue(const Axis&);
    DictionaryValue(const std::wstring&);
    DictionaryValue(const RubyCNTK::Dictionary&);
    DictionaryValue(const DictionaryValue&);

    bool HasValue();
    enum Type ValueType();

    void Save(const std::wstring& filename);
    static DictionaryValue Load(const std::wstring& filename);

    %extend{
      bool __eq__(const DictionaryValue& other) {
        return (*$self) == other;
      }

      bool Value_bool__() {
        return $self->Value<bool>();
      }

      int Value_int__() {
        return $self->Value<int>();
      }

      size_t Value_size_t__() {
        return $self->Value<size_t>();
      }

      float Value_float__() {
        return $self->Value<float>();
      }

      double Value_double__() {
        return $self->Value<double>();
      }

      Axis& Value_axis__() {
        return $self->Value<RubyCNTK::Axis>();
      }

      std::wstring& Value_wstring__() {
        return $self->Value<std::wstring>();
      }

      std::vector<RubyCNTK::DictionaryValue>& Value_vec_dict_value__() {
        return $self->Value<std::vector<RubyCNTK::DictionaryValue> >();
      }

      RubyCNTK::Dictionary& Value_dict__() {
        return $self->Value<RubyCNTK::Dictionary>();
      }

      RubyCNTK::NDArrayView& Value_ndarrayview__() {
        return $self->Value<RubyCNTK::NDArrayView>();
      }

    }
  };

  class Dictionary {
  public:
    Dictionary();
    ~Dictionary();

    Dictionary(const Dictionary&);
    bool Contains(const std::wstring& key);
    void Add(const Dictionary& other);
    size_t Size();
    void Save(const std::wstring& filename);
    static Dictionary Load(const std::wstring& filename);

    %extend{
      DictionaryValue& __getitem__(const std::wstring& key) {
        return (*$self)[key];
      }

      void __setitem__(const std::wstring& key, const DictionaryValue v) {
        (*$self)[key] = v;
      }

      bool __eq__(const Dictionary& other) {
        return (*$self) == other;
      }
    }
  };

  enum class VariableKind {
    Input = 0,
      Output = 1,
      Parameter = 2,
      Constant = 3,
      Placeholder = 4,
   };
  const wchar_t* VariableKindName(VariableKind variableKind);



  class Variable {
  public:
    Variable(const FunctionPtr& function);
    Variable(const NDShape& shape, bool isSparse, enum DataType dataType, bool needsGradient, const std::wstring& name, const std::vector<Axis>& dynamicAxes, const std::wstring& uid);

    ~Variable();
    const NDShape& Shape();
    const std::vector<Axis>& DynamicAxes();
    VariableKind Kind();
    bool IsSparse();
    bool IsInput();
    bool IsOutput();
    bool IsParameter();
    bool IsConstant();
    bool IsPlaceholder();
    const std::wstring& Name();
    const std::wstring& Uid();
    FunctionPtr Owner();
    enum DataType GetDataType();
    bool NeedsGradient();

    %extend{
      RubyCNTK::FunctionPtr to_function() {
        return *$self;
      }

      bool __eq__(const Variable& other) {
        return (*$self) == other;
      }

      FunctionPtr __neg__() {
        return -(*$self);
      }

    }
  };


  // REMOVE
  //  Variable PlaceholderVariable(const NDShape& shape, enum DataType dataType, const std::wstring& name, const std::vector<Axis>& dynamicAxes);
  Variable PlaceholderVariable(const NDShape& shape, const std::wstring& name, const std::vector<Axis>& dynamicAxes);
  //  Variable PlaceholderVariable(const NDShape& shape, const std::vector<Axis>& dynamicAxes = Axis::UnknownDynamicAxes());
  //  Variable PlaceholderVariable(const std::wstring& name = L"");
  Variable InputVariable(const NDShape& shape, bool isSparse, enum DataType dataType, 
                         bool needsGradient, const std::wstring& name, 
                         const std::vector<Axis>& dynamicAxes = Axis::DefaultInputVariableDynamicAxes());

  // REMOVE
  // Variable InputVariable(const NDShape& shape, enum DataType dataType, bool needsGradient, const std::wstring& name = L"", const std::vector<Axis>& dynamicAxes = Axis::DefaultInputVariableDynamicAxes());
  //   Variable InputVariable(const NDShape& shape, enum DataType dataType, const std::wstring& name, const std::vector<Axis>& dynamicAxes = Axis::DefaultInputVariableDynamicAxes());
  //  Variable InputVariable(const NDShape& shape, enum DataType dataType, const std::vector<Axis>& dynamicAxes = Axis::DefaultInputVariableDynamicAxes());
  //  Variable InputVariable(const NDShape& shape, bool isSparse, enum DataType dataType, const std::wstring& name, const std::vector<Axis>& dynamicAxes = Axis::DefaultInputVariableDynamicAxes());
  //  Variable InputVariable(const NDShape& shape, bool isSparse, enum DataType dataType, const std::vector<Axis>& dynamicAxes = Axis::DefaultInputVariableDynamicAxes());
  
  Variable OutputVariable(const NDShape& shape, enum DataType dataType, const std::vector<Axis>& dynamicAxes, const std::wstring& name /*= L""*/);

  static const int SentinelValueForInferParamInitRank = std::numeric_limits<int>::max();
  static const int DefaultParamInitScale = 1;
  static const int DefaultParamInitOutputRank = 1;
  static const int DefaultParamInitFilterRank = 0;

  Dictionary ConstantInitializer(double value = 0.0);
  Dictionary UniformInitializer(double scale, unsigned long seed = SentinelValueForAutoSelectRandomSeed);
  Dictionary NormalInitializer(double scale, int outputRank = SentinelValueForInferParamInitRank, int filterRank = SentinelValueForInferParamInitRank, unsigned long seed = SentinelValueForAutoSelectRandomSeed);
  Dictionary XavierInitializer(double scale = DefaultParamInitScale, int outputRank = SentinelValueForInferParamInitRank, int filterRank = SentinelValueForInferParamInitRank, unsigned long seed = SentinelValueForAutoSelectRandomSeed);
  Dictionary GlorotUniformInitializer(double scale = DefaultParamInitScale, int outputRank = SentinelValueForInferParamInitRank, int filterRank = SentinelValueForInferParamInitRank, unsigned long seed = SentinelValueForAutoSelectRandomSeed);
  Dictionary GlorotNormalInitializer(double scale = DefaultParamInitScale, int outputRank = SentinelValueForInferParamInitRank, int filterRank = SentinelValueForInferParamInitRank, unsigned long seed = SentinelValueForAutoSelectRandomSeed);
  Dictionary HeUniformInitializer(double scale = DefaultParamInitScale, int outputRank = SentinelValueForInferParamInitRank, int filterRank = SentinelValueForInferParamInitRank, unsigned long seed = SentinelValueForAutoSelectRandomSeed);
  Dictionary HeNormalInitializer(double scale = DefaultParamInitScale, int outputRank = SentinelValueForInferParamInitRank, int filterRank = SentinelValueForInferParamInitRank, unsigned long seed = SentinelValueForAutoSelectRandomSeed);
  Dictionary BilinearInitializer(size_t kernelWidth, size_t kernelHeight);
  Dictionary RandomInitializerWithRank(const Dictionary& initializer, int outputRank, int filterRank);

  class Parameter : public Variable {
  public:
    Parameter(const Variable& variable);
    Parameter(const NDArrayViewPtr& value, const std::wstring& name = L"");

    // REMOVE
    // template<typename ElemType>
    // Parameter(const NDShape& shape, ElemType initValue, const DeviceDescriptor& device = DeviceDescriptor::UseDefaultDevice(), const std::wstring& name = L"");

    Parameter(const NDShape& shape, enum DataType dataType, double initValue, const DeviceDescriptor& device = DeviceDescriptor::UseDefaultDevice(), const std::wstring& name = L"");
    Parameter(const NDShape& shape, enum DataType dataType, const Dictionary& initializer, const DeviceDescriptor& device = DeviceDescriptor::UseDefaultDevice(), const std::wstring& name = L"");
    ~Parameter();

    size_t CurrentValueTimeStamp();
    void RecordValueUpdate();
    NDArrayViewPtr Value();
    void SetValue(const NDArrayViewPtr& value);

  };

  // REMOVE
  // %template(ParameterFloat) RubyCNTK::Parameter::Parameter<float>;
  // %template(ParameterDouble) RubyCNTK::Parameter::Parameter<double>;

  class Constant : public Variable {
  public:
    Constant(const Variable& variable);
    Constant(const NDArrayViewPtr& value, const std::wstring& name = L"");

    template<typename ElemType>
    Constant(const NDShape& shape, ElemType initValue, const DeviceDescriptor& device = DeviceDescriptor::UseDefaultDevice(), const std::wstring& name = L"");

    Constant(const NDShape& shape, enum DataType dataType, double initValue, const DeviceDescriptor& device = DeviceDescriptor::UseDefaultDevice(), const std::wstring& name = L"");
    static Constant Scalar(enum DataType dataType, double value, const ::CNTK::DeviceDescriptor& device = DeviceDescriptor::CPUDevice());
    
    ~Constant();

    NDArrayViewPtr Value();


  };

%template(ConstantFloat) RubyCNTK::Constant::Constant<float>;
%template(ConstantDouble) RubyCNTK::Constant::Constant<double>;

  class Value {
  public:
    Value(const NDArrayViewPtr& data);
    Value(const NDArrayViewPtr& data, const NDMaskPtr& mask);
    
    template <typename ElementType>
    static ValuePtr Create(const NDShape& sampleShape, const std::vector<std::vector<ElementType>>& sequences, const std::vector<bool>& sequenceStartFlags, const DeviceDescriptor& device, bool readOnly = false);
    template <typename ElementType>
    static ValuePtr Create(const NDShape& sampleShape, const std::vector<std::vector<ElementType>>& sequences, const DeviceDescriptor& device, bool readOnly = false);

    ~Value();
    //    DeviceDescriptor Device();
    enum DataType GetDataType();
    enum StorageFormat GetStorageFormat();
    NDShape& Shape();
    bool IsSparse();
    bool IsReadOnly();
    size_t MaskedCount();
    NDArrayViewPtr Data();
    NDMaskPtr Mask();
    ValuePtr DeepClone(bool readOnly);
    ValuePtr DeepClone();
    ValuePtr Alias(bool readOnly = false);
    void CopyFrom(const Value& source);
    void CopyVariableValueTo(const Variable& outputVariable, std::vector<std::vector<size_t>>& sequences);
    


  };

  %nodefaultctor BackPropState;
  class BackPropState {
  public:
    //    BackPropState(const FunctionPtr&, const DeviceDescriptor&, const std::unordered_map<Variable, ValuePtr>& forwardPropValuesToSave = {});
    ~BackPropState();

    FunctionPtr Function();
    //    DeviceDescriptor Device();
    std::unordered_map<Variable, ValuePtr>& SavedForwardPropValues();
    
  };
  enum class ParameterCloningMethod {
    Share, Clone, Freeze,
  };

  
  %nodefaultctor Function;
  class Function {
  public:
    ~Function();
    BackPropStatePtr Forward(const std::unordered_map<Variable, ValuePtr>& arguments,
                             std::unordered_map<Variable, ValuePtr>& outputs,
                             const DeviceDescriptor& computeDevice = DeviceDescriptor::UseDefaultDevice(),
                             const std::unordered_set<Variable>& outputsToRetainBackwardStateFor = {});
    void Backward(const BackPropStatePtr& state,
                  const std::unordered_map<Variable, ValuePtr>& rootGradientValues,
                  std::unordered_map<Variable, ValuePtr>& backPropagatedGradientValuesForInputs);
    const std::wstring& OpName();

    void Evaluate(const std::unordered_map<Variable, ValuePtr>& arguments,
                  std::unordered_map<Variable, ValuePtr>& outputs,
                  const DeviceDescriptor& computeDevice = DeviceDescriptor::UseDefaultDevice());
    FunctionPtr Clone(ParameterCloningMethod parameterCloneMethod = ParameterCloningMethod::Clone, const std::unordered_map<Variable, Variable>& replacements = {});

    size_t CurrentVersion();
    std::wstring& Name();
    void SetName(const std::wstring& name);

    std::wstring& Uid();
    FunctionPtr RootFunction();
    bool IsComposite();
    bool IsPrimitive();
    bool IsBlock();
    FunctionPtr BlockRoot();
    std::vector<std::pair<Variable, Variable> > BlockArgumentsMapping();
    std::vector<Variable> Inputs();
    Variable Output();
    std::vector<Variable> Outputs();
    std::vector<Variable> Arguments();
    std::vector<Parameter> Parameters();
    std::vector<Constant> Constants();
    std::vector<Variable> Placeholders();
    const Dictionary& Attributes();
    FunctionPtr ReplacePlaceholders(const std::unordered_map<Variable, Variable>&);
    FunctionPtr ReplacePlaceholder(const Variable&);
    void SaveModel(const std::wstring& modelFile);
    void RestoreModel(const std::wstring& modelFilePath);
    static FunctionPtr LoadModel(const std::wstring& modelFile, const DeviceDescriptor& computeDevice = DeviceDescriptor::UseDefaultDevice());
    void PrintGraph();


    Dictionary Serialize();
    static FunctionPtr Deserialize(const Dictionary& dictionary, const ::CNTK::DeviceDescriptor& device = DeviceDescriptor::UseDefaultDevice());
  };


  FunctionPtr Negate(const Variable& operand, const std::wstring& name = L"");
  FunctionPtr Sigmoid(const Variable& operand, const std::wstring& name = L"");
  FunctionPtr Tanh(const Variable& operand, const std::wstring& name = L"");
  FunctionPtr Sin(const Variable& operand, const std::wstring& name = L"");
  FunctionPtr Cos(const Variable& operand, const std::wstring& name = L"");
  FunctionPtr ReLU(const Variable& operand, const std::wstring& name = L"");
  FunctionPtr Exp(const Variable& operand, const std::wstring& name = L"");
  FunctionPtr Log(const Variable& operand, const std::wstring& name = L"");
  FunctionPtr Square(const Variable& operand, const std::wstring& name = L"");
  FunctionPtr Sqrt(const Variable& operand, const std::wstring& name = L"");
  FunctionPtr Round(const Variable& operand, const std::wstring& name = L"");
  FunctionPtr Floor(const Variable& operand, const std::wstring& name = L"");
  FunctionPtr Ceil(const Variable& operand, const std::wstring& name = L"");
  FunctionPtr Abs(const Variable& operand, const std::wstring& name = L"");
  FunctionPtr Reciprocal(const Variable& operand, const std::wstring& name = L"");
  FunctionPtr Softmax(const Variable& operand, const std::wstring& name = L"");
  FunctionPtr Hardmax(const Variable& operand, const std::wstring& name = L"");
  FunctionPtr TransposeAxes(const Variable& operand, const Axis& axis1, const Axis& axis2, const std::wstring& name = L"");
  FunctionPtr Transpose(const Variable& operand, const std::wstring& name = L"");
  FunctionPtr Slice(const Variable& operand, const Axis& axis, int beginIndex, int endIndex, const std::wstring& name = L"");
  FunctionPtr RandomSample(const Variable& operand, size_t numSamples, bool allowDuplicates, const std::wstring& name /*= L""*/);
  FunctionPtr RandomSampleInclusionFrequency(const Variable& operand, size_t numSamples, bool allowDuplicates, const std::wstring& name /*= L""*/);
  FunctionPtr Dropout(const Variable& operand, double dropoutRate, const std::wstring& name = L"");
  FunctionPtr Reshape(const Variable& operand, const NDShape& replacementShape, const Axis& beginAxis, const Axis& endAxis, const std::wstring& name = L"");
  FunctionPtr Reshape(const Variable& operand, const NDShape& newShape, const std::wstring& name = L"");
  FunctionPtr Plus(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name = L"");
  FunctionPtr Minus(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name = L"");
  FunctionPtr LogAddExp(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name = L"");
  FunctionPtr ElementTimes(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name = L"");
  FunctionPtr ElementDivide(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name = L"");
  FunctionPtr Equal(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name = L"");
  FunctionPtr NotEqual(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name = L"");
  FunctionPtr Less(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name = L"");
  FunctionPtr LessEqual(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name = L"");
  FunctionPtr Greater(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name = L"");
  FunctionPtr GreaterEqual(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name = L"");
  FunctionPtr Times(const Variable& leftOperand, const Variable& rightOperand, size_t outputRank, int inferInputRankToMap, const std::wstring& name = L"");
  FunctionPtr Times(const Variable& leftOperand, const Variable& rightOperand, size_t outputRank, const std::wstring& name = L"");
  FunctionPtr Times(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name = L"");
  FunctionPtr TransposeTimes(const Variable& leftOperand, const Variable& rightOperand, size_t outputRank, const std::wstring& name = L"");
  FunctionPtr TransposeTimes(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name = L"");
  
  FunctionPtr CosineDistance(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name = L"");
  FunctionPtr BinaryCrossEntropy(const Variable& prediction, const Variable& targets, const std::wstring& name = L"");
  FunctionPtr WeightedBinaryCrossEntropy(const Variable& prediction, const Variable& targets, const Variable& weights, const std::wstring& name = L"");
  FunctionPtr SquaredError(const Variable& prediction, const Variable& targets, const std::wstring& name = L"");
  FunctionPtr CrossEntropyWithSoftmax(const Variable& prediction, const Variable& labels, const Axis& axis, const std::wstring& name = L"");
  FunctionPtr CrossEntropyWithSoftmax(const Variable& prediction, const Variable& labels, const std::wstring& name = L"");
  FunctionPtr ClassificationError(const Variable& prediction, const Variable& labels, size_t topN, const Axis& axis, const std::wstring& name = L"");
  FunctionPtr ClassificationError(const Variable& prediction, const Variable& labels, const Axis& axis, const std::wstring& name = L"");
  FunctionPtr ClassificationError(const Variable& prediction, const Variable& labels, const std::wstring& name = L"");
  FunctionPtr LambdaRank(const Variable& prediction, const Variable& gains, const Variable& groupId, const std::wstring& name = L"");
  FunctionPtr NDCGAt1(const Variable& prediction, const Variable& gains, const Variable& groupId, const std::wstring& name = L"");
 

  /// *** important functions *** 
  FunctionPtr PastValue(const Variable& operand, const Variable& initialState, size_t offset = 1, const std::wstring& name = L"");
  FunctionPtr PastValue(const Variable& operand, size_t offset = 1, const std::wstring& name = L"");

  FunctionPtr FutureValue(const Variable& operand, const Variable& initialState, size_t offset = 1, const std::wstring& name = L"");
  FunctionPtr FutureValue(const Variable& operand, size_t offset = 1, const std::wstring& name = L"");


  FunctionPtr ReduceSum(const Variable& operand, const std::wstring& name = L"");
  FunctionPtr ReduceSum(const Variable& operand, const Axis& axis, const std::wstring& name = L"");
  FunctionPtr ReduceLogSum(const Variable& operand, const Axis& axis, const std::wstring& name = L"");
  FunctionPtr ReduceMean(const Variable& operand, const Axis& axis, const std::wstring& name = L"");
  FunctionPtr ReduceMax(const Variable& operand, const Axis& axis, const std::wstring& name = L"");
  FunctionPtr ReduceMin(const Variable& operand, const Axis& axis, const std::wstring& name = L"");
  FunctionPtr PerDimMeanVarianceNormalize(const Variable& operand, const NDArrayViewPtr& mean, const NDArrayViewPtr& invStdDev, const std::wstring& name = L"");
  
  FunctionPtr Convolution(const Variable& convolutionMap,
                          const Variable& operand,
                          const NDShape& strides = {1},
                          const std::vector<bool>& sharing = {true},
                          const std::vector<bool>& autoPadding = {true},
                          const NDShape& lowerPad = {0},
                          const NDShape& upperPad = {0},
                          bool transpose = false,
                          size_t maxTempMemSizeInSamples = 0,
                          const std::wstring& name = L"");

  FunctionPtr ROIPooling(const Variable& convolutionMap, const Variable& rois, const NDShape& roiOutputShape, const std::wstring& name = L"");

  enum class PoolingType { Max, Average, };

  FunctionPtr Pooling(const Variable& operand,
                      PoolingType poolingType,
                      const NDShape& poolingWindowShape,
                      const NDShape& strides = {1},
                      const std::vector<bool>& autoPadding = {false},
                      const NDShape& lowerPad = {0},
                      const NDShape& upperPad = {0},
                      const std::wstring& name = L"");

  FunctionPtr Unpooling(const Variable& operand,
                        const Variable& poolingInput,
                        PoolingType UnpoolingType,
                        const NDShape& UnpoolingWindowShape,
                        const NDShape& strides = { 1 },
                        const std::vector<bool>& autoPadding = { false },
                        const NDShape& lowerPad = { 0 },
                        const NDShape& upperPad = { 0 },
                        const std::wstring& name = L"");

  FunctionPtr BatchNormalization(const Variable& operand,
                                 const Variable& scale,
                                 const Variable& bias,
                                 const Variable& runningMean,
                                 const Variable& runningInvStd,
                                 const Variable& runningSampleCount,
                                 bool spatial,
                                 double normalizationTimeConstant = 0,
                                 double blendTimeConstant = 0,
                                 double epsilon = 0.00001,
                                 bool useCuDNNEngine = true,
                                 const std::wstring& name = L"");

  FunctionPtr OptimizedRNNStack(const Variable& operand, const Variable& weights, size_t hiddenSize, size_t numLayers, bool bidirectional = false, const std::wstring& recurrentOp = L"lstm", const std::wstring& name = L"");
  FunctionPtr Clip(const Variable& operand, const Variable& min, const Variable& max, const std::wstring& name = L"");
  FunctionPtr ElementSelect(const Variable& condition, const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name = L"");
  FunctionPtr Splice(const std::vector<Variable>& operands, const Axis& axis, const std::wstring& name = L"");
  FunctionPtr Combine(const std::vector<Variable>& operands, const std::wstring& name = L"");
  FunctionPtr Alias(const Variable& operand, const std::wstring& name = L"");
  FunctionPtr AsBlock(FunctionPtr&& composite, const std::vector<std::pair<Variable, Variable>>& argumentsMap, const std::wstring& blockOpName, const std::wstring& blockName = L"");
  
    namespace Sequence
    {
      FunctionPtr IsFirst(const Variable& operand, const std::wstring& name = L"");
      FunctionPtr IsLast(const Variable& operand, const std::wstring& name = L"");
      
      //      FunctionPtr Slice(const Variable& operand, int beginIndex, int endIndex, const std::wstring& name = L"");
      
      ///
      /// Create an instance of the CNTK built-in sum reduction operation on specified tensor input operand along the operands lone dynamic sequence axis
      ///
      //      FunctionPtr ReduceSum(const Variable& operand, const std::wstring& name = L"");
      
      FunctionPtr First(const Variable& operand, const std::wstring& name = L"");
      FunctionPtr Last(const Variable& operand, const std::wstring& name = L"");
      
      FunctionPtr Where(const Variable& condition, const std::wstring& name = L"");
      FunctionPtr Gather(const Variable& operand, const Variable& condition, const std::wstring& name = L"");
      FunctionPtr Scatter(const Variable& operand, const Variable& condition, const std::wstring& name = L"");
      
      FunctionPtr BroadcastAs(const Variable& operand, const Variable& broadcastAs, const std::wstring& name = L"");
    }

  template <typename T>
  class TrainingParameterSchedule
  {
  public:
    enum class UnitType : unsigned int
    { Sample = 0, Minibatch = 1, };
    //    static const size_t EntireSweep = 0;

    TrainingParameterSchedule(T value, UnitType unit);
    TrainingParameterSchedule(const std::vector<T>& schedule, UnitType unit, size_t epochSize = 1);
    TrainingParameterSchedule(const std::vector<std::pair<size_t, T>>& schedule, UnitType unit, size_t epochSize = 1);
    
  };
    
  %template(LearningRateSchedule) TrainingParameterSchedule<double>;

  template <typename T, typename TrainingParameterSchedule<T>::UnitType U>
  class TrainingParameterPerUnitSchedule : public TrainingParameterSchedule<T>
  {
  public:
    TrainingParameterPerUnitSchedule(double value);
    TrainingParameterPerUnitSchedule(const std::vector<double>& schedule, size_t epochSize = 1);    
    TrainingParameterPerUnitSchedule(const std::vector<std::pair<size_t, double>>& schedule, size_t epochSize = 1);
    
    const double __getitem__(size_t count);
    };

  %template(TrainingParameterPerSampleSchedule) TrainingParameterPerUnitSchedule<double, CNTK::TrainingParameterSchedule<double>::UnitType::Sample>;
  
  typedef TrainingParameterSchedule<double> MomentumSchedule;

  struct AdditionalLearningOptions
  {
    double l1RegularizationWeight = 0.0;
    double l2RegularizationWeight = 0.0;
    TrainingParameterPerUnitSchedule<double, TrainingParameterSchedule<double>::UnitType::Minibatch> gaussianNoiseInjectionStdDev = 0.0;
    double gradientClippingThresholdPerSample = std::numeric_limits<double>::infinity();
    bool gradientClippingWithTruncation = true;
  };

  //  bool DefaultUnitGainValue();
  //  void SetDefaultUnitGainValue(bool value);

  %nodefaultctor Learner;
  class Learner {
  public:
    virtual bool Update(std::unordered_map<Parameter, NDArrayViewPtr>& gradientValues, size_t trainingSampleCount) = 0;
    virtual const std::vector<Parameter>& Parameters();
    virtual Dictionary CreateCheckpoint();
    virtual void RestoreFromCheckpoint(const Dictionary&);
    virtual ~Learner();
    virtual void ResetLearningRate(const RubyCNTK::LearningRateSchedule& learningRateSchedule);
    virtual void ResetSmoothedGradients() = 0;
    virtual double LearningRate();
    size_t TotalNumberOfSamplesSeen();
  };

  LearnerPtr SGDLearner(const std::vector<Parameter>& parameters,
                        const RubyCNTK::LearningRateSchedule& learningRateSchedule,
                        AdditionalLearningOptions additionalOptions = AdditionalLearningOptions());

  LearnerPtr MomentumSGDLearner(const std::vector<Parameter>& parameters,
                                const RubyCNTK::LearningRateSchedule& learningRateSchedule,
                                const MomentumSchedule& momentumSchedule,
                                bool unitGain = DefaultUnitGainValue(),
                                AdditionalLearningOptions additionalOptions = AdditionalLearningOptions());

  LearnerPtr NesterovLearner(const std::vector<Parameter>& parameters,
                             const RubyCNTK::LearningRateSchedule& learningRateSchedule,
                             const MomentumSchedule& momentumSchedule,
                             bool unitGain = DefaultUnitGainValue(),
                             AdditionalLearningOptions additionalOptions = AdditionalLearningOptions());

  static MomentumSchedule DefaultVarianceMomentum = MomentumAsTimeConstantSchedule(2 * 3600 * 100);

  LearnerPtr AdamLearner(const std::vector<Parameter>& parameters,
                         const RubyCNTK::LearningRateSchedule& learningRateSchedule,
                         const MomentumSchedule& momentumSchedule,
                         bool unitGain = DefaultUnitGainValue(),
                         const MomentumSchedule& varianceMomentumSchedule = DefaultVarianceMomentum,
                         bool lowMemory = true,
                         AdditionalLearningOptions additionalOptions = AdditionalLearningOptions());

  LearnerPtr RMSPropLearner(const std::vector<Parameter>& parameters,
                            const RubyCNTK::LearningRateSchedule& learningRateSchedule,
                            double gamma,
                            double inc,
                            double dec,
                            double max,
                            double min,
                            bool needAveMultiplier = true,
                            AdditionalLearningOptions additionalOptions = AdditionalLearningOptions());
  
  %nodefaultctor DistributedLearner;
   class DistributedLearner : public Learner
    {
    public:

       virtual DistributedCommunicatorPtr GetCommunicator();


      bool Update(std::unordered_map<RubyCNTK::Parameter, RubyCNTK::NDArrayViewPtr>& gradientValues, size_t minibatchSampleCount, bool sweepEnd = false) override;
      virtual void ResetLearningRate(const RubyCNTK::LearningRateSchedule& learningRateSchedule);
      virtual double LearningRate();
      void ResetSmoothedGradients() override;
      virtual size_t ParallelizationAfter();
      virtual bool Update(std::unordered_map<RubyCNTK::Parameter, RubyCNTK::NDArrayViewPtr>& gradientValues,
                          MinibatchInfo& minibatch);
    };

  //  RubyCNTK::DistributedLearnerPtr CreateDataParallelDistributedLearner(DistributedCommunicatorPtr communicator, LearnerPtr learner, size_t distributeAfterSamples, bool useAsyncBufferedParameterUpdate = false);
  //  RubyCNTK::DistributedLearnerPtr CreateQuantizedDataParallelDistributedLearner(QuantizedDistributedCommunicatorPtr communicator, LearnerPtr learner, size_t distributeAfterSamples, bool useAsyncBufferedParameterUpdate = false);


  struct StreamInformation {
    std::wstring m_name;
    size_t m_id;
    enum StorageFormat m_storageFormat;
    enum DataType m_elementType;
    NDShape m_sampleLayout;

    %extend {
      bool __eq__(const RubyCNTK::StreamInformation& other) const {
        return (*$self) == other;
      }
    }
  };

  struct MinibatchData;

  %nodefaultctor Trainer;
  class Trainer {
  public:

    bool TrainMinibatch(const std::unordered_map<Variable, MinibatchData>& arguments, const DeviceDescriptor& computeDevice = DeviceDescriptor::UseDefaultDevice());
    bool TrainMinibatch(const std::unordered_map<Variable, ValuePtr>& arguments, const DeviceDescriptor& computeDevice = DeviceDescriptor::UseDefaultDevice());
    bool TrainMinibatch(const std::unordered_map<Variable, MinibatchData>& arguments, std::unordered_map<Variable, ValuePtr>& outputsToFetch, const DeviceDescriptor& computeDevice = DeviceDescriptor::UseDefaultDevice());
    bool TrainMinibatch(const std::unordered_map<Variable, ValuePtr>& arguments, std::unordered_map<Variable, ValuePtr>& outputsToFetch, const DeviceDescriptor& computeDevice = DeviceDescriptor::UseDefaultDevice());
    double TestMinibatch(const std::unordered_map<Variable, MinibatchData>& arguments, const DeviceDescriptor& computeDevice = DeviceDescriptor::UseDefaultDevice());
    double TestMinibatch(const std::unordered_map<Variable, ValuePtr>& arguments, const DeviceDescriptor& computeDevice = DeviceDescriptor::UseDefaultDevice());

    void SaveCheckpoint(const std::wstring& filePath, Dictionary externalState = Dictionary());
    Dictionary RestoreFromCheckpoint(const std::wstring& filePath);
    
    FunctionPtr Model();
    FunctionPtr LossFunction();
    FunctionPtr EvaluationFunction();
    double PreviousMinibatchLossAverage() const;
    double PreviousMinibatchEvaluationAverage() const;
    size_t PreviousMinibatchSampleCount();
    std::vector<LearnerPtr>& ParameterLearners() const;
    size_t TotalNumberOfSamplesSeen() const;

  };
  
  TrainerPtr CreateTrainer(const FunctionPtr& model, const FunctionPtr& lossFunction, const std::vector<LearnerPtr>& parameterLearners);
  TrainerPtr CreateTrainer(const FunctionPtr& model, const FunctionPtr& lossFunction, const FunctionPtr& evaluationFunction, const std::vector<LearnerPtr>& parameterLearners);

  struct MinibatchData {
    MinibatchData();
    MinibatchData(ValuePtr value);
    MinibatchData(ValuePtr value, size_t numSamples, bool sweepEnd = false);
    MinibatchData(ValuePtr value, size_t numSequences, size_t numSamples, bool sweepEnd);

    ValuePtr data;
    size_t numberOfSequences;
    size_t numberOfSamples;
    bool sweepEnd;
  };

  %nodefaultctor MinibatchSource;
  class MinibatchSource {
  public:
    static const size_t InfinitelyRepeat = SIZE_MAX;
    static const size_t FullDataSweep    = SIZE_MAX - 2; // An arbitrary sentinel value
    static const size_t InfiniteSamples  = SIZE_MAX;
    static const size_t DefaultRandomizationWindow = SIZE_MAX - 2;

    const std::unordered_set<RubyCNTK::StreamInformation>& StreamInfos() = 0;
    
  };

  MinibatchSourcePtr CreateCompositeMinibatchSource(const Dictionary& configuration);

  struct StreamConfiguration
  {
    StreamConfiguration(const std::wstring& streamName, size_t dim, bool isSparse = false, const std::wstring& streamAlias = L"");

    std::wstring m_streamName;
    size_t m_dim;
    bool m_isSparse;
    std::wstring m_streamAlias;
  };

  MinibatchSourcePtr TextFormatMinibatchSource(const std::wstring& dataFilePath, 
                                               const std::vector<StreamConfiguration>& streamConfigs,
                                               size_t epochSize = MinibatchSource::InfinitelyRepeat, 
                                               bool randomize = true,
                                               size_t randomizationWindow = MinibatchSource::DefaultRandomizationWindowInChunks,
                                               bool sampleBasedRandomizationWindow = false);
  void ComputeInputPerDimMeansAndInvStdDevs(const MinibatchSourcePtr& minibatchSource,
                                            std::unordered_map<StreamInformation, 
                                            std::pair<NDArrayViewPtr, NDArrayViewPtr>>& computedMeanAndVariances,
                                            const DeviceDescriptor& device = DeviceDescriptor::CPUDevice());
  void SetMaxNumCPUThreads(size_t numCPUThreads);
  size_t GetMaxNumCPUThreads();

  struct DistributedWorkerDescriptor {
    size_t m_globalRank;
    std::wstring m_hostId;
    bool IsMain();

    %extend{
      bool __eq__(const RubyCNTK::DistributedWorkerDescriptor& other) {
        return (*$self) == other;
      }
    }
  };

  class DistributedCommunicator
    {
    public:
      virtual const std::unordered_set<DistributedWorkerDescriptor>& Workers() const = 0;
      virtual const DistributedWorkerDescriptor& CurrentWorker() const = 0;
      virtual DistributedCommunicatorPtr SubGroup(const std::unordered_set<DistributedWorkerDescriptor>& subGroupWorkers) const = 0;
      virtual void Concatenate(
                               const std::vector<ValuePtr>& values,
                               std::vector<ValuePtr>& outputValues,
                               const std::unordered_set<DistributedWorkerDescriptor>& sendToWorkers) = 0;
      virtual void Concatenate(
                               const std::vector<NDArrayViewPtr>& input,
                               std::vector<NDArrayViewPtr>& output,
                               const std::unordered_set<DistributedWorkerDescriptor>& sendToWorkers) = 0;
      
      virtual void Gather(
                          const Dictionary& input,
                          std::vector<DictionaryPtr>& output,
                          const std::unordered_set<DistributedWorkerDescriptor>& sendToWorkers) = 0;
      
      virtual void AggregateInPlace(
                                    const std::vector<NDArrayViewPtr>& values,
                                    const std::unordered_set<DistributedWorkerDescriptor>& sendToWorkers) = 0;
      
      virtual void Aggregate(
                             const std::vector<NDArrayViewPtr>& values,
                             std::vector<NDArrayViewPtr>& outputValues,
                             const std::unordered_set<DistributedWorkerDescriptor>& sendToWorkers) = 0;
      
      virtual ~DistributedCommunicator() {}
      static void Finalize();
      virtual void Barrier() = 0;
  };

  class QuantizedDistributedCommunicator : public DistributedCommunicator {
  public:
    virtual void QuantizedAggregate(
                                    const std::vector<NDArrayViewPtr>& inValues,
                                    const std::vector<NDArrayViewPtr>& valueQuantizationResidues,
                                    const std::vector<NDArrayViewPtr>& stripeQuantizationResidues,
                                    std::vector<NDArrayViewPtr>& aggregatedOutputs,
                                    std::vector<NDArrayViewPtr>& newQuantizationResidues,
                                    std::vector<NDArrayViewPtr>& newStripeQuantizationResidues,
                                    const std::unordered_set<DistributedWorkerDescriptor>& sendToWorkers) = 0;

    virtual void QuantizedAggregateInPlace(
                                           std::vector<NDArrayViewPtr>& inValues,
                                           std::vector<NDArrayViewPtr>& valueQuantizationResidues,
                                           std::vector<NDArrayViewPtr>& stripeQuantizationResidues,
                                           const std::unordered_set<DistributedWorkerDescriptor>& sendToWorkers) = 0;
  };

  DistributedCommunicatorPtr MPICommunicator();
  QuantizedDistributedCommunicatorPtr QuantizedMPICommunicator(bool zeroThresholdFor1Bit, bool useQuantizationForSelfStripe, size_t numQuantizationBits);

  typedef TrainingParameterPerUnitSchedule<size_t, TrainingParameterSchedule<size_t>::UnitType::Sample> MinibatchSizeSchedule;

  class TrainingSession {
    struct PeriodicAction
    {
      size_t frequency;
      size_t currentIndex;
      size_t sampleCountWhenLastCalled;
      std::function<void(size_t currentIndex, const DeviceDescriptor&)> action;
    };
  public:
    TrainingSession(
                    const MinibatchSourcePtr& trainingSource,
                    const TrainerPtr& trainer,
                    const std::unordered_map<Variable, StreamInformation>& modelInputToMinibatchSourceStream,
                    const MinibatchSizeSchedule& minibatchSizeSchedule,
                    size_t checkpointFrequencyInSamples,
                    const std::wstring& checkPointFileName,
                    const MinibatchSourcePtr& crossValidationSource = nullptr,
                    const MinibatchSizeSchedule& crossValidationSchedule = MinibatchSizeSchedule(1),
                    size_t crossValidationFrequencyInSamples = std::numeric_limits<size_t>::max(),
                    bool restoreFromCheckpointIfExists = true,
                    bool keepExistingCheckpoints = false,
                    size_t maxNumberOfTrainingSamples = std::numeric_limits<size_t>::max(),
                    size_t progressFrequency = std::numeric_limits<size_t>::max());

    void Train(const DeviceDescriptor& computeDevice);

    void RestoreFromCheckpoint(const std::wstring& checkpointFileName);
    virtual ~TrainingSession();
    virtual size_t GetMinibatchSize();
    virtual void OnMinibatchStart();
    virtual void OnMinibatchEnd();
    virtual void OnCheckpointStart(size_t /*checkpointIndex*/);
    virtual void OnCheckpointEnd(size_t /*checkpointIndex*/);
    virtual void OnCrossValidationStart(size_t /*validationIndex*/) {};
    virtual void OnCrossValidationEnd(size_t /*validationIndex*/, double /*averageError*/, size_t /*numberOfSamples*/, size_t /*numberOfMinibatches*/);
    virtual void OnProgress(size_t /*index*/);
  };

  TrainingSessionPtr 
  CreateBasicTrainingSession(
                             const MinibatchSourcePtr& trainingSource,
                             const TrainerPtr& trainer,
                             const std::unordered_map<Variable, StreamInformation>& modelInputToMinibatchSourceStream,
                             const MinibatchSizeSchedule& minibatchSizeSchedule,
                             size_t checkpointFrequencyInSamples,
                             const std::wstring& checkPointFileName,
                             const MinibatchSourcePtr& crossValidationSource = nullptr,
                             const MinibatchSizeSchedule& crossValidationSchedule = MinibatchSizeSchedule(1),
                             size_t crossValidationFrequencyInSamples = std::numeric_limits<size_t>::max(),
                             bool restoreFromCheckpointIfExists = true,
                             bool keepExistingCheckpoints = false,
                             size_t maxNumberOfTrainingSamples = std::numeric_limits<size_t>::max(),
                             size_t progressFrequency = std::numeric_limits<size_t>::max());

};
