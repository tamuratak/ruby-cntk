
%ignore CNTK::TrainingParameterSchedule::operator=;
%ignore CNTK::TrainingParameterSchedule::operator[];

  template <typename T>
  class TrainingParameterSchedule
  {
  public:
    enum class UnitType : unsigned int
    { Sample = 0, Minibatch = 1, }; 
    //    static const size_t EntireSweep = 0;
    
    TrainingParameterSchedule(T value, CNTK::TrainingParameterSchedule<T>::UnitType unit);
    TrainingParameterSchedule(const std::vector<T>& schedule, CNTK::TrainingParameterSchedule<T>::UnitType unit, size_t epochSize = 1);
    TrainingParameterSchedule(const std::vector<std::pair<size_t, T>>& schedule, CNTK::TrainingParameterSchedule<T>::UnitType unit, size_t epochSize = 1);
  };

  %template(MomentumSchedule)   CNTK::TrainingParameterSchedule<double>;
  typedef CNTK::TrainingParameterSchedule<double>  LearningRateSchedule;

  template <typename T, typename U>
  class TrainingParameterPerUnitSchedule : public TrainingParameterSchedule<T>
  {
  public:
    TrainingParameterPerUnitSchedule(T value);
    TrainingParameterPerUnitSchedule(const std::vector<T>& schedule, 
                                     size_t epochSize = TrainingParameterSchedule<T>::FullDataSweep);
    TrainingParameterPerUnitSchedule(const std::vector<std::pair<size_t, T>>& schedule, 
                                     size_t epochSize = TrainingParameterSchedule<T>::FullDataSweep);
  };

  typedef TrainingParameterPerUnitSchedule<size_t, TrainingParameterSchedule<size_t>::UnitType::Sample> MinibatchSizeSchedule;

%rename(TrainingParameterPerSampleSchedule) TrainingParameterPerUnitSchedule<double, CNTK::TrainingParameterSchedule<double>::UnitType::Sample>;
  class TrainingParameterPerUnitSchedule<double, CNTK::TrainingParameterSchedule<double>::UnitType::Sample> : public TrainingParameterSchedule<double>
  {
  public:
    TrainingParameterPerUnitSchedule(double value);
    TrainingParameterPerUnitSchedule(const std::vector<double>& schedule, size_t epochSize = 1);
    TrainingParameterPerUnitSchedule(const std::vector<std::pair<size_t, double>>& schedule, size_t epochSize = 1);
    const double __getitem__(size_t count);
  };

%rename(TrainingParameterPerMinibatchSchedule) TrainingParameterPerUnitSchedule<double, CNTK::TrainingParameterSchedule<double>::UnitType::Minibatch>;
class TrainingParameterPerUnitSchedule<double, CNTK::TrainingParameterSchedule<double>::UnitType::Minibatch>  : public TrainingParameterSchedule<double>
  {
  public:
    TrainingParameterPerUnitSchedule(double value);
    TrainingParameterPerUnitSchedule(const std::vector<double>& schedule, size_t epochSize = 1);
    TrainingParameterPerUnitSchedule(const std::vector<std::pair<size_t, double>>& schedule, size_t epochSize = 1);
    const double __getitem__(size_t count);
  };

  class MomentumAsTimeConstantSchedule: public TrainingParameterSchedule<double>
  {
  public:
    MomentumAsTimeConstantSchedule(double value);
    MomentumAsTimeConstantSchedule(const std::vector<double>& schedule, size_t epochSize = FullDataSweep);
    MomentumAsTimeConstantSchedule(const std::vector<std::pair<size_t, double> >& schedule, size_t epochSize = FullDataSweep);
  };



  struct AdditionalLearningOptions
  {
    double l1RegularizationWeight = 0.0;
    double l2RegularizationWeight = 0.0;
    CNTK::TrainingParameterPerUnitSchedule<double, CNTK::TrainingParameterSchedule<double>::UnitType::Minibatch> gaussianNoiseInjectionStdDev = 0.0;
    double gradientClippingThresholdPerSample = std::numeric_limits<double>::infinity();
    bool gradientClippingWithTruncation = true;
  };

  bool DefaultUnitGainValue();
  void SetDefaultUnitGainValue(bool value);

  %nodefaultctor Learner;
  class Learner {
  public:
    virtual bool Update(std::unordered_map<Parameter, NDArrayViewPtr>& gradientValues, size_t trainingSampleCount) = 0;
    virtual const std::vector<Parameter>& Parameters();
    virtual Dictionary CreateCheckpoint();
    virtual void RestoreFromCheckpoint(const Dictionary&);
    virtual ~Learner();
    virtual void ResetLearningRate(const CNTK::LearningRateSchedule& learningRateSchedule);
    virtual void ResetSmoothedGradients() = 0;
    virtual double LearningRate();
    size_t TotalNumberOfSamplesSeen();
  };

  LearnerPtr SGDLearner(const std::vector<Parameter>& parameters,
                        const CNTK::LearningRateSchedule& learningRateSchedule,
                        AdditionalLearningOptions additionalOptions = AdditionalLearningOptions());

  LearnerPtr MomentumSGDLearner(const std::vector<Parameter>& parameters,
                                const CNTK::LearningRateSchedule& learningRateSchedule,
                                const CNTK::MomentumSchedule& momentumSchedule,
                                bool unitGain = DefaultUnitGainValue(),
                                AdditionalLearningOptions additionalOptions = AdditionalLearningOptions());

  LearnerPtr NesterovLearner(const std::vector<Parameter>& parameters,
                             const CNTK::LearningRateSchedule& learningRateSchedule,
                             const CNTK::MomentumSchedule& momentumSchedule,
                             bool unitGain = DefaultUnitGainValue(),
                             AdditionalLearningOptions additionalOptions = AdditionalLearningOptions());

  static CNTK::MomentumSchedule DefaultVarianceMomentum = MomentumAsTimeConstantSchedule(2 * 3600 * 100);

  LearnerPtr AdamLearner(const std::vector<Parameter>& parameters,
                         const CNTK::LearningRateSchedule& learningRateSchedule,
                         const CNTK::MomentumSchedule& momentumSchedule,
                         bool unitGain = DefaultUnitGainValue(),
                         const CNTK::MomentumSchedule& varianceMomentumSchedule = DefaultVarianceMomentum,
                         AdditionalLearningOptions additionalOptions = AdditionalLearningOptions());

  LearnerPtr RMSPropLearner(const std::vector<Parameter>& parameters,
                            const CNTK::LearningRateSchedule& learningRateSchedule,
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
