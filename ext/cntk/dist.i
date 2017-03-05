  %nodefaultctor DistributedLearner;
   class DistributedLearner : public Learner
    {
    public:

       virtual DistributedCommunicatorPtr GetCommunicator();


      bool Update(std::unordered_map<CNTK::Parameter, CNTK::NDArrayViewPtr>& gradientValues, size_t minibatchSampleCount, bool sweepEnd = false) override;
      virtual void ResetLearningRate(const CNTK::LearningRateSchedule& learningRateSchedule);
      virtual double LearningRate();
      void ResetSmoothedGradients() override;
      virtual size_t ParallelizationAfter();
      virtual bool Update(std::unordered_map<CNTK::Parameter, CNTK::NDArrayViewPtr>& gradientValues,
                          MinibatchInfo& minibatch);
    };

  struct DistributedWorkerDescriptor {
    size_t m_globalRank;
    std::wstring m_hostId;
    bool IsMain();

    %extend{
      bool __eq__(const CNTK::DistributedWorkerDescriptor& other) {
        return (*$self) == other;
      }
    }
  };

  //  CNTK::DistributedLearnerPtr CreateDataParallelDistributedLearner(DistributedCommunicatorPtr communicator, LearnerPtr learner, size_t distributeAfterSamples, bool useAsyncBufferedParameterUpdate = false);
  //  CNTK::DistributedLearnerPtr CreateQuantizedDataParallelDistributedLearner(QuantizedDistributedCommunicatorPtr communicator, LearnerPtr learner, size_t distributeAfterSamples, bool useAsyncBufferedParameterUpdate = false);
  

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

