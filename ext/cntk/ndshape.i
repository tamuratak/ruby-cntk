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
