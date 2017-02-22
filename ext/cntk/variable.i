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

  Variable PlaceholderVariable(const NDShape& shape, const std::wstring& name, const std::vector<Axis>& dynamicAxes);
  Variable InputVariable(const NDShape& shape, bool isSparse, enum DataType dataType, 
                         bool needsGradient, const std::wstring& name, 
                         const std::vector<Axis>& dynamicAxes = Axis::DefaultInputVariableDynamicAxes());
  Variable OutputVariable(const NDShape& shape, enum DataType dataType, const std::vector<Axis>& dynamicAxes, const std::wstring& name /*= L""*/);

  static const int SentinelValueForInferParamInitRank = std::numeric_limits<int>::max();
  static const int DefaultParamInitScale = 1;
  static const int DefaultParamInitOutputRank = 1;
  static const int DefaultParamInitFilterRank = 0;

  Dictionary ConstantInitializer(double value = 0.0);
  Dictionary UniformInitializer(double scale, unsigned long seed = SentinelValueForAutoSelectRandomSeed);
  Dictionary NormalInitializer(double scale, int outputRank = SentinelValueForInferParamInitRank, 
                               int filterRank = SentinelValueForInferParamInitRank, unsigned long seed = SentinelValueForAutoSelectRandomSeed);
  Dictionary XavierInitializer(double scale = DefaultParamInitScale, int outputRank = SentinelValueForInferParamInitRank,
                               int filterRank = SentinelValueForInferParamInitRank, unsigned long seed = SentinelValueForAutoSelectRandomSeed);
  Dictionary GlorotUniformInitializer(double scale = DefaultParamInitScale, int outputRank = SentinelValueForInferParamInitRank,
                                      int filterRank = SentinelValueForInferParamInitRank, unsigned long seed = SentinelValueForAutoSelectRandomSeed);
  Dictionary GlorotNormalInitializer(double scale = DefaultParamInitScale, int outputRank = SentinelValueForInferParamInitRank,
                                     int filterRank = SentinelValueForInferParamInitRank, unsigned long seed = SentinelValueForAutoSelectRandomSeed);
  Dictionary HeUniformInitializer(double scale = DefaultParamInitScale, int outputRank = SentinelValueForInferParamInitRank,
                                  int filterRank = SentinelValueForInferParamInitRank, unsigned long seed = SentinelValueForAutoSelectRandomSeed);
  Dictionary HeNormalInitializer(double scale = DefaultParamInitScale, int outputRank = SentinelValueForInferParamInitRank,
                                 int filterRank = SentinelValueForInferParamInitRank, unsigned long seed = SentinelValueForAutoSelectRandomSeed);
  Dictionary BilinearInitializer(size_t kernelWidth, size_t kernelHeight);
  Dictionary RandomInitializerWithRank(const Dictionary& initializer, int outputRank, int filterRank);

  class Parameter : public Variable {
  public:
    Parameter(const Variable& variable);
    Parameter(const NDArrayViewPtr& value, const std::wstring& name = L"");

    Parameter(const NDShape& shape, enum DataType dataType, double initValue, const DeviceDescriptor& device = DeviceDescriptor::UseDefaultDevice(), const std::wstring& name = L"");
    Parameter(const NDShape& shape, enum DataType dataType, const Dictionary& initializer, const DeviceDescriptor& device = DeviceDescriptor::UseDefaultDevice(), const std::wstring& name = L"");
    ~Parameter();

    size_t CurrentValueTimeStamp();
    void RecordValueUpdate();
    NDArrayViewPtr Value();
    void SetValue(const NDArrayViewPtr& value);

  };

  class Constant : public Variable {
  public:
    Constant(const Variable& variable);
    Constant(const NDArrayViewPtr& value, const std::wstring& name = L"");

    Constant(const NDShape& shape, enum DataType dataType, double initValue, const DeviceDescriptor& device = DeviceDescriptor::UseDefaultDevice(), const std::wstring& name = L"");
    static Constant Scalar(enum DataType dataType, double value, const ::CNTK::DeviceDescriptor& device = DeviceDescriptor::CPUDevice());
    
    ~Constant();

    NDArrayViewPtr Value();


  };

  class Value {
  public:
    Value(const NDArrayViewPtr& data);
    Value(const NDArrayViewPtr& data, const NDMaskPtr& mask);
    
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
