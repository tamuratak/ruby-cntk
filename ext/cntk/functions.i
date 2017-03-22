
  %nodefaultctor BackPropState;
  class BackPropState {
  public:
    //    BackPropState(const FunctionPtr& function, const DeviceDescriptor& computeDevice, const std::unordered_map<Variable, ValuePtr>& forwardPropValuesToSave = {});
    ~BackPropState();

    FunctionPtr Function();
    std::unordered_map<Variable, ValuePtr>& SavedForwardPropValues();

    %extend {
      %newobject device;
      DeviceDescriptor* device() {
        return new CNTK::DeviceDescriptor((*$self).Device());
      }
    }
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

    void Gradients(const std::unordered_map<Variable, ValuePtr>& arguments,
                   std::unordered_map<Variable, ValuePtr>& gradients,
                   const DeviceDescriptor& computeDevice = DeviceDescriptor::UseDefaultDevice());
    void Evaluate(const std::unordered_map<Variable, ValuePtr>& arguments,
                  std::unordered_map<Variable, ValuePtr>& outputs,
                  const DeviceDescriptor& computeDevice = DeviceDescriptor::UseDefaultDevice());
    FunctionPtr Clone(ParameterCloningMethod parameterCloneMethod = ParameterCloningMethod::Clone, const std::unordered_map<Variable, Variable>& replacements = {});

    FunctionPtr FindByName(const std::wstring& name, bool nestedSearchInsideBlockFunction = false);
    std::vector<FunctionPtr> FindAllWithName(const std::wstring& name, bool nestedSearchInsideBlockFunction = false);

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
    std::wstring AsString(bool doNotInferOutputs = true);
    static const int MaxNumOutputs = 64;

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
  FunctionPtr CosineDistanceWithNegativeSamples(const Variable& leftOperand, const Variable& rightOperand, size_t shiftWindow, size_t numberOfNegativeSamples, const std::wstring& name = L"");
  FunctionPtr BinaryCrossEntropy(const Variable& prediction, const Variable& targets, const std::wstring& name = L"");
  FunctionPtr WeightedBinaryCrossEntropy(const Variable& prediction, const Variable& targets, const Variable& weights, const std::wstring& name = L"");
  FunctionPtr SquaredError(const Variable& prediction, const Variable& targets, const std::wstring& name = L"");
  FunctionPtr CrossEntropyWithSoftmax(const Variable& prediction, const Variable& labels, const Axis& axis, const std::wstring& name = L"");
  FunctionPtr CrossEntropyWithSoftmax(const Variable& prediction, const Variable& labels, const std::wstring& name = L"");
  FunctionPtr EditDistanceError(const Variable& prediction, const Variable& labels, float substitutionPenalty, float deletionPenalty, float insertionPenalty, bool squashInputs, const std::vector<size_t>& samplesToIgnore, const std::wstring& name = L"");
  FunctionPtr ForwardBackward(const Variable& graph, const Variable& features, size_t blankTokenId, int delayConstraint, const std::wstring& name = L"");
  FunctionPtr LabelsToGraph(const Variable& labels, const std::wstring& name = L"");
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
  FunctionPtr ReduceProd(const Variable& operand, const Axis& axis, const std::wstring& name = L"");
  FunctionPtr PerDimMeanVarianceNormalize(const Variable& operand, const NDArrayViewPtr& mean, const NDArrayViewPtr& invStdDev, const std::wstring& name = L"");
  
  FunctionPtr Convolution(const Variable& convolutionMap,
                          const Variable& operand,
                          const NDShape& strides = {1},
                          const std::vector<bool>& sharing = {true},
                          const std::vector<bool>& autoPadding = {true},
                          const NDShape& lowerPad = {0},
                          const NDShape& upperPad = {0},
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
                      const bool ceilOutDim = false,
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
  FunctionPtr StopGradient(const Variable& operand, const std::wstring& name = L"");
  FunctionPtr AsComposite(const FunctionPtr& rootFunction, const std::wstring& name = L"");
  FunctionPtr ELU(const Variable& operand, const std::wstring& name = L"");
  FunctionPtr LeakyReLU(const Variable& operand, const std::wstring& name = L"");
  FunctionPtr PReLU(const Variable& alpha, const Variable& operand, const std::wstring& name = L"");
  FunctionPtr Softplus(const Variable& operand, const std::wstring& name = L"");
  FunctionPtr Argmax(const Variable& operand, const Axis& axis, const std::wstring& name = L"");
  FunctionPtr Argmin(const Variable& operand, const Axis& axis, const std::wstring& name = L"");

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

