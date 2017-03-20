
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
                    const CNTK::MinibatchSizeSchedule& minibatchSizeSchedule,
                    size_t checkpointFrequencyInSamples,
                    const std::wstring& checkPointFileName,
                    const MinibatchSourcePtr& crossValidationSource = nullptr,
                    const CNTK::MinibatchSizeSchedule& crossValidationSchedule = CNTK::MinibatchSizeSchedule(1),
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
  };

  TrainingSessionPtr 
  CreateBasicTrainingSession(
                             const MinibatchSourcePtr& trainingSource,
                             const TrainerPtr& trainer,
                             const std::unordered_map<Variable, StreamInformation>& modelInputToMinibatchSourceStream,
                             const CNTK::MinibatchSizeSchedule& minibatchSizeSchedule,
                             size_t checkpointFrequencyInSamples,
                             const std::wstring& checkPointFileName,
                             const MinibatchSourcePtr& crossValidationSource = nullptr,
                             const CNTK::MinibatchSizeSchedule& crossValidationSchedule = CNTK::MinibatchSizeSchedule(1),
                             size_t crossValidationFrequencyInSamples = std::numeric_limits<size_t>::max(),
                             bool restoreFromCheckpointIfExists = true,
                             bool keepExistingCheckpoints = false,
                             size_t maxNumberOfTrainingSamples = std::numeric_limits<size_t>::max(),
                             size_t progressFrequency = std::numeric_limits<size_t>::max());
