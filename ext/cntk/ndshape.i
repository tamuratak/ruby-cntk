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

    static const NDShape Unknown;

    %extend {
      bool __eq__(const NDShape& other) {
        return (*$self) == other;
      }
    }
  };

  enum class MaskKind : char
  {Invalid = 0, Valid = 1, SequenceBegin = 2,};

  class NDMask
  {
  public:
    NDMask(const NDShape& shape, const DeviceDescriptor& device = DeviceDescriptor::CPUDevice());
    ~NDMask();

    void InvalidateSection(const std::vector<size_t>& sectionOffset, const NDShape& sectionShape);
    void MarkSequenceBegin(const std::vector<size_t>& offset);
    void MarkSequenceBegin(const std::vector<size_t>& offset, const NDShape& sectionShape);
    void Clear();
    size_t MaskedCount();
    //    DeviceDescriptor Device() const { return m_device; }
    const NDShape& Shape();
    const MaskKind* DataBuffer();
    NDMaskPtr DeepClone(const DeviceDescriptor& device);
    NDMaskPtr DeepClone();
    NDMaskPtr Alias();
    void CopyFrom(const NDMask& source);
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
