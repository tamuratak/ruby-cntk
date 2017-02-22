
  template <typename T>
  class TrainingParameterSchedule
  {
  public:
    enum class UnitType : unsigned int
    { Sample = 0, Minibatch = 1, }; 
    //    static const size_t EntireSweep = 0;
    
    TrainingParameterSchedule(T value, RubyCNTK::TrainingParameterSchedule<T>::UnitType unit);
    TrainingParameterSchedule(const std::vector<T>& schedule, RubyCNTK::TrainingParameterSchedule<T>::UnitType unit, size_t epochSize = 1);
    TrainingParameterSchedule(const std::vector<std::pair<size_t, T>>& schedule, RubyCNTK::TrainingParameterSchedule<T>::UnitType unit, size_t epochSize = 1);
    
  };
  
  %template(MomentumSchedule) RubyCNTK::TrainingParameterSchedule<double>;
  
  %rename(MinibatchSizeSchedule) TrainingParameterPerUnitSchedule<double, RubyCNTK::TrainingParameterSchedule<double>::UnitType::Sample>;  
  class TrainingParameterPerUnitSchedule<double, RubyCNTK::TrainingParameterSchedule<double>::UnitType::Sample> // : public TrainingParameterSchedule<double>
  {
  public:
    TrainingParameterPerUnitSchedule(double value);
    TrainingParameterPerUnitSchedule(const std::vector<double>& schedule, size_t epochSize = 1);
    TrainingParameterPerUnitSchedule(const std::vector<std::pair<size_t, double>>& schedule, size_t epochSize = 1);
    
    const double __getitem__(size_t count);
  };
  
  %rename(TrainingParameterPerMinibatchSchedule) TrainingParameterPerUnitSchedule<double, RubyCNTK::TrainingParameterSchedule<double>::UnitType::Minibatch>;
  class TrainingParameterPerUnitSchedule<double, RubyCNTK::TrainingParameterSchedule<double>::UnitType::Minibatch> // : public TrainingParameterSchedule<double>
  {
  public:
    TrainingParameterPerUnitSchedule(double value);
    TrainingParameterPerUnitSchedule(const std::vector<double>& schedule, size_t epochSize = 1);
    TrainingParameterPerUnitSchedule(const std::vector<std::pair<size_t, double>>& schedule, size_t epochSize = 1);

    const double __getitem__(size_t count);
  };

  struct AdditionalLearningOptions
  {
    double l1RegularizationWeight = 0.0;
    double l2RegularizationWeight = 0.0;
    RubyCNTK::TrainingParameterPerUnitSchedule<double, RubyCNTK::TrainingParameterSchedule<double>::UnitType::Minibatch> gaussianNoiseInjectionStdDev = 0.0;
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
                                const RubyCNTK::MomentumSchedule& momentumSchedule,
                                bool unitGain = DefaultUnitGainValue(),
                                AdditionalLearningOptions additionalOptions = AdditionalLearningOptions());

  LearnerPtr NesterovLearner(const std::vector<Parameter>& parameters,
                             const RubyCNTK::LearningRateSchedule& learningRateSchedule,
                             const RubyCNTK::MomentumSchedule& momentumSchedule,
                             bool unitGain = DefaultUnitGainValue(),
                             AdditionalLearningOptions additionalOptions = AdditionalLearningOptions());

  static RubyCNTK::MomentumSchedule DefaultVarianceMomentum = MomentumAsTimeConstantSchedule(2 * 3600 * 100);

  LearnerPtr AdamLearner(const std::vector<Parameter>& parameters,
                         const RubyCNTK::LearningRateSchedule& learningRateSchedule,
                         const RubyCNTK::MomentumSchedule& momentumSchedule,
                         bool unitGain = DefaultUnitGainValue(),
                         const RubyCNTK::MomentumSchedule& varianceMomentumSchedule = DefaultVarianceMomentum,
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
  
  struct MinibatchData;
  %nodefaultctor Trainer;
  class Trainer {
  public:

    bool TrainMinibatch(const std::unordered_map<Variable, MinibatchData>& arguments, const DeviceDescriptor& computeDevice = DeviceDescriptor::UseDefaultDevice());
    bool TrainMinibatch(const std::unordered_map<Variable, ValuePtr>& arguments, const DeviceDescriptor& computeDevice = DeviceDescriptor::UseDefaultDevice());
    bool TrainMinibatch(const std::unordered_map<Variable, MinibatchData>& arguments, std::unordered_map<Variable, ValuePtr>& outputsToFetch, 
                        const DeviceDescriptor& computeDevice = DeviceDescriptor::UseDefaultDevice());
    bool TrainMinibatch(const std::unordered_map<Variable, ValuePtr>& arguments, std::unordered_map<Variable, ValuePtr>& outputsToFetch, 
                        const DeviceDescriptor& computeDevice = DeviceDescriptor::UseDefaultDevice());
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
