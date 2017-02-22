
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
