
  %rename(name) StreamInformation::m_name;
  %rename(id)  StreamInformation::m_id;
  %rename(storage_format) StreamInformation::m_storageFormat;
  %rename(element_type) StreamInformation::m_elementType;
  %rename(sample_layout) StreamInformation::m_sampleLayout;

  struct StreamInformation {
    std::wstring m_name;
    size_t m_id;
    enum StorageFormat m_storageFormat;
    enum DataType m_elementType;
    NDShape m_sampleLayout;

    %extend {
      bool __eq__(const CNTK::StreamInformation& other) const {
        return (*$self) == other;
      }
    }
  };

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

    virtual ~MinibatchSource();

    const std::unordered_set<CNTK::StreamInformation>& StreamInfos() = 0;

    const std::unordered_map<StreamInformation, MinibatchData>& 
    GetNextMinibatch(size_t minibatchSizeInSamples, 
                     const DeviceDescriptor& device = DeviceDescriptor::UseDefaultDevice());
    
    const std::unordered_map<StreamInformation, MinibatchData>& 
    GetNextMinibatch(size_t minibatchSizeInSequences,
                     size_t minibatchSizeInSamples,
                     const DeviceDescriptor& device = DeviceDescriptor::UseDefaultDevice());

    virtual const std::unordered_map<StreamInformation, MinibatchData>& 
    GetNextMinibatch(size_t minibatchSizeInSequences,
                     size_t minibatchSizeInSamples,
                     size_t numberOfWorkers,
                     size_t workerRank,
                     const DeviceDescriptor& device = DeviceDescriptor::UseDefaultDevice()) = 0;

    virtual Dictionary GetCheckpointState() const;
    virtual void RestoreFromCheckpoint(const Dictionary& /*checkpoint*/);
    const StreamInformation& StreamInfo(const std::wstring& streamName);
    const StreamInformation& StreamInfo(const Variable& variableToMatch);
  };

  MinibatchSourcePtr CreateCompositeMinibatchSource(const Dictionary& configuration);

