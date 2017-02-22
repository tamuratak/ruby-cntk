%module CNTK
#pragma SWIG nowarn=801

 // 
 // The file is based on the followings
 // https://github.com/Microsoft/CNTK/blob/master/bindings/python/cntk/cntk_py.i
 // https://github.com/Microsoft/CNTK/blob/master/Source/CNTKv2LibraryDll/API/CNTKLibrary.h
 //

%include std_wstring.i
%include std_vector.i
%include std_map.i
%include std_unordered_map.i
%include std_unordered_set.i
%include std_pair.i
%include std_shared_ptr.i

%template(StdVectorSizeT) std::vector<size_t>;
%template(StdVectorDouble) std::vector<double>;
%template(StdVectorFloat) std::vector<float>;
%template(StdVectorVectorSizeT) std::vector<std::vector<size_t> >;
%template(StdVectorVectorFloat) std::vector<std::vector<float> >;
%template(StdVectorVectorDouble) std::vector<std::vector<double> >;

%shared_ptr(RubyCNTK::Trainer)
%shared_ptr(RubyCNTK::TrainingSession)
%shared_ptr(RubyCNTK::BasicTrainingSession)
%shared_ptr(RubyCNTK::Function)
%shared_ptr(RubyCNTK::NDArrayView)
%shared_ptr(RubyCNTK::Value)
%shared_ptr(RubyCNTK::NDMask)
%shared_ptr(RubyCNTK::BackPropState)
%shared_ptr(RubyCNTK::Learner)
%shared_ptr(RubyCNTK::MinibatchSource)
%shared_ptr(RubyCNTK::DistributedCommunicator)
%shared_ptr(RubyCNTK::QuantizedDistributedCommunicator)
%shared_ptr(RubyCNTK::DistributedLearner)


%{
#include <vector>
#include <CNTKLibrary.h>
%}
%inline %{
namespace CNTK {};
namespace RubyCNTK {
  using namespace CNTK;
  static RubyCNTK::DeviceDescriptor __cpu_device__ = RubyCNTK::DeviceDescriptor::CPUDevice();
  static RubyCNTK::DeviceDescriptor __best_device__ = RubyCNTK::DeviceDescriptor::BestDevice();
  static std::vector<RubyCNTK::DeviceDescriptor> __all_device__;
};
%}

namespace RubyCNTK {
  typedef std::shared_ptr<RubyCNTK::NDArrayView> NDArrayViewPtr;
  typedef std::shared_ptr<RubyCNTK::Dictionary> DictionaryPtr;
  typedef std::shared_ptr<RubyCNTK::Function> FunctionPtr;
  typedef std::shared_ptr<RubyCNTK::Value> ValuePtr;
  typedef std::shared_ptr<RubyCNTK::NDMask> NDMaskPtr;
  typedef std::shared_ptr<RubyCNTK::BackPropState> BackPropStatePtr;
  typedef std::shared_ptr<RubyCNTK::Learner> LearnerPtr;
  typedef std::shared_ptr<RubyCNTK::DistributedCommunicator> DistributedCommunicatorPtr;
  typedef std::shared_ptr<RubyCNTK::QuantizedDistributedCommunicator> QuantizedDistributedCommunicatorPtr;
  typedef std::shared_ptr<RubyCNTK::Trainer> TrainerPtr;
  typedef std::shared_ptr<RubyCNTK::MinibatchSource> MinibatchSourcePtr;
  typedef std::shared_ptr<RubyCNTK::TrainingSession> TrainingSessionPtr;
};


%template() std::vector<RubyCNTK::DeviceDescriptor>;
%template(StdVectorVariable) std::vector<RubyCNTK::Variable>;
%template(StdVectorPairVariableVariable) std::vector< std::pair<RubyCNTK::Variable, RubyCNTK::Variable> >;
%template(StdVectorDictionaryValue) std::vector< RubyCNTK::DictionaryValue >;
%template(StdUMapVariableValue) std::unordered_map< RubyCNTK::Variable, RubyCNTK::ValuePtr >;
%template(StdUMapVariablevariable) std::unordered_map< RubyCNTK::Variable, RubyCNTK::Variable >;
%template(StdUSetVariable) std::unordered_set<RubyCNTK::Variable>;
%template(StdUSetDistributedWorkerDescriptor) std::unordered_set<RubyCNTK::DistributedWorkerDescriptor>;
%template(StdUMapStreamInfoMinibatchData) std::unordered_map<RubyCNTK::StreamInformation, RubyCNTK::MinibatchData>;

///************************************
/// renaming rule
///
///************************************

%rename("__%(utitle)s__", %$isfunction, notregexmatch$name="Initializer|Learner$") "";
%rename("%(utitle)s", %$isfunction, regexmatch$name="Initializer|Learner$") "";
%rename("%(utitle)s", %$ismember, %$isfunction) "";
%rename("%(utitle)s", %$ismember, %$isvariable) "";
%rename("%s", %$isenum) "";
%rename("%s", %$isconstructor) "";
%rename(__forward__) RubyCNTK::Function::Forward;

%typecheck(1000) RubyCNTK::NDShape const &, RubyCNTK::NDShape {
    // '1000' is the typecheck precedence code. It means: check after basic
    // types, but before arrays. See: http://www.swig.org/Doc3.0/Typemaps.html#Typemaps_overloading
  $1 = NIL_P(rb_check_array_type($input)) ? 0 : 1;
}

%typemap(in) const RubyCNTK::NDShape& (RubyCNTK::NDShape tmp) {
  VALUE arry = rb_check_array_type($input);
  if(NIL_P(arry)) {
    rb_raise(rb_eArgError, "Array expected"); SWIG_fail;
  }else{
    std::vector<size_t> dimensions(RARRAY_LEN(arry));
    for (int i=0; i<RARRAY_LEN(arry); i++) {
	VALUE elt = RARRAY_AREF(arry, i);
        dimensions[i] = NUM2SIZET(elt);
    }
    tmp = CNTK::NDShape(dimensions);
    $1 = &tmp;
  }
}

//
// Exception handling
//
%exception {
    try { $action }
    catch (const std::runtime_error &e) { SWIG_exception(SWIG_RuntimeError,e.what()); }
    catch (const std::invalid_argument &e) { SWIG_exception(SWIG_ValueError,e.what()); }
    catch (const std::logic_error &e) { SWIG_exception(SWIG_RuntimeError,e.what()); }
    catch (...) { SWIG_exception(SWIG_UnknownError,"Runtime exception"); }
}

//
// In the future, we would just swtich to %include "CNTKLibrary.h". 
//  



namespace RubyCNTK {
  enum class DataType {
    Unknown,
    Float,
    Double
  };

  const char* DataTypeName(enum DataType);
  size_t DataTypeSize(enum DataType);

  enum class StorageFormat
  {
    Dense,
    SparseCSC,
    SparseBlockCol,
  };

  enum class DeviceKind
  {
    CPU,
    GPU,
  };

  struct MinibatchInfo
  {
    bool atEndOfData;
    //    bool atEndOfSweep;
    size_t numberOfSamples;
    NDArrayViewPtr trainingLossValue;
    NDArrayViewPtr evalCriterionValue;
    
    bool IsEmpty();
  };

  %nodefaultctor DeviceDescriptor;
  class DeviceDescriptor 
  {
  public:
    unsigned int Id();
    RubyCNTK::DeviceKind Type();

    %extend{

      %newobject CPUDevice;
      static RubyCNTK::DeviceDescriptor* CPUDevice(){
        return new RubyCNTK::DeviceDescriptor(RubyCNTK::DeviceDescriptor::CPUDevice());
      }

      %newobject GPUDevice;
      static RubyCNTK::DeviceDescriptor* GPUDevice(unsigned int deviceId){
        return new RubyCNTK::DeviceDescriptor(RubyCNTK::DeviceDescriptor::GPUDevice(deviceId));
      }

      %newobject DefaultDevice;
      static RubyCNTK::DeviceDescriptor* DefaultDevice(){
        return new RubyCNTK::DeviceDescriptor(RubyCNTK::DeviceDescriptor::DefaultDevice());
      }

      %newobject UseDefaultDevice;
      static RubyCNTK::DeviceDescriptor* UseDefaultDevice(){
        return new RubyCNTK::DeviceDescriptor(RubyCNTK::DeviceDescriptor::UseDefaultDevice());
      }

      %newobject BestDevice;
      static RubyCNTK::DeviceDescriptor* BestDevice(){
        return new RubyCNTK::DeviceDescriptor(RubyCNTK::DeviceDescriptor::BestDevice());
      }

      bool __eq__(const DeviceDescriptor& other){
        return (*$self) == other;
      }

    }

    static void SetDefaultDevice(const DeviceDescriptor& newDefaultDevice);
    static const std::vector<DeviceDescriptor>& AllDevices();

  };

  %include "ndshape.i"
  %include "ndarrayview.i"
  %include "dictionary.i"
  %include "variable.i"
  %include "functions.i"
  %include "trainer.i"
  %include "io.i"

  void ComputeInputPerDimMeansAndInvStdDevs(const MinibatchSourcePtr& minibatchSource,
                                            std::unordered_map<StreamInformation, 
                                            std::pair<NDArrayViewPtr, NDArrayViewPtr>>& computedMeanAndVariances,
                                            const DeviceDescriptor& device = DeviceDescriptor::CPUDevice());
  void SetMaxNumCPUThreads(size_t numCPUThreads);
  size_t GetMaxNumCPUThreads();

  %include "dist.i"
  %include "session.i"

};
