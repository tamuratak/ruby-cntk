%module CNTK
#pragma SWIG nowarn=801

 // 
 // The file is based on the followings
 // https://github.com/Microsoft/CNTK/blob/master/bindings/python/cntk/cntk_py.i
 // https://github.com/Microsoft/CNTK/blob/master/Source/CNTKv2LibraryDll/API/CNTKLibrary.h
 //

%{
#include <vector>
#include <memory>
#include <CNTKLibrary.h>
%}

%include std_wstring.i
%include std_vector.i
%include std_map.i
%include std_unordered_map.i
%include std_unordered_set.i
%include std_pair.i
%include std_shared_ptr.i


%template() std::vector<bool>;
%template() std::vector<char>;
%template() std::vector<int>;
%template() std::vector<size_t>;
%template() std::vector<float>;
%template() std::vector<double>;
%template() std::vector<std::vector<size_t>>;
%template() std::vector<std::vector<float>>;
%template() std::vector<std::vector<double>>;


%shared_ptr(CNTK::Trainer)
%shared_ptr(CNTK::TrainingSession)
%shared_ptr(CNTK::BasicTrainingSession)
%shared_ptr(CNTK::Function)
%shared_ptr(CNTK::NDArrayView)
%shared_ptr(CNTK::Value)
%shared_ptr(CNTK::NDMask)
%shared_ptr(CNTK::BackPropState)
%shared_ptr(CNTK::Learner)
%shared_ptr(CNTK::MinibatchSource)
%shared_ptr(CNTK::DistributedCommunicator)
%shared_ptr(CNTK::QuantizedDistributedCommunicator)
%shared_ptr(CNTK::DistributedLearner)



%inline %{
namespace CNTK {};
%}

namespace CNTK {
  typedef std::shared_ptr<CNTK::NDArrayView> NDArrayViewPtr;
  typedef std::shared_ptr<CNTK::Dictionary> DictionaryPtr;
  typedef std::shared_ptr<CNTK::Function> FunctionPtr;
  typedef std::shared_ptr<CNTK::Value> ValuePtr;
  typedef std::shared_ptr<CNTK::NDMask> NDMaskPtr;
  typedef std::shared_ptr<CNTK::BackPropState> BackPropStatePtr;
  typedef std::shared_ptr<CNTK::Learner> LearnerPtr;
  typedef std::shared_ptr<CNTK::DistributedCommunicator> DistributedCommunicatorPtr;
  typedef std::shared_ptr<CNTK::QuantizedDistributedCommunicator> QuantizedDistributedCommunicatorPtr;
  typedef std::shared_ptr<CNTK::Trainer> TrainerPtr;
  typedef std::shared_ptr<CNTK::MinibatchSource> MinibatchSourcePtr;
  typedef std::shared_ptr<CNTK::TrainingSession> TrainingSessionPtr;
};


%template() std::vector<CNTK::Variable>;
%template() std::vector<CNTK::OutputVariable>;
%template() std::vector<CNTK::Parameter>;
%template() std::vector<CNTK::Constant>;
%template() std::vector<CNTK::Axis>;
%template() std::vector<CNTK::DeviceDescriptor>;
%template() std::vector<CNTK::StreamConfiguration>;
%template() std::vector<std::shared_ptr<CNTK::NDArrayView>>;
%template() std::vector<std::shared_ptr<CNTK::Value>>;
%template() std::vector<std::shared_ptr<CNTK::Function>>;
%template() std::vector<std::shared_ptr<CNTK::Learner>>;
%template() std::vector<std::shared_ptr<CNTK::DistributedLearner>>;
%template() std::vector<std::shared_ptr<CNTK::Trainer>>;

// %template() std::vector<CNTK::Parameter>;
%template(StdVectorPairVariableVariable)      std::vector< std::pair<CNTK::Variable, CNTK::Variable> >;
%template(StdVectorDictionaryValue)           std::vector< CNTK::DictionaryValue >;
%template(StdUMapVariableValue)               std::unordered_map< CNTK::Variable, CNTK::ValuePtr >;
%template()                                   std::unordered_map< CNTK::Variable, CNTK::Variable >;
%template()                                   std::unordered_map< CNTK::Variable, CNTK::MinibatchData>;
%template()                                   std::unordered_map< CNTK::Parameter, CNTK::NDArrayViewPtr>;
%template(StdUSetVariable)                    std::unordered_set<CNTK::Variable>;
%template(StdUSetDistributedWorkerDescriptor) std::unordered_set<CNTK::DistributedWorkerDescriptor>;
%template(MinibatchTable)                     std::unordered_map<CNTK::StreamInformation, CNTK::MinibatchData>;

///************************************
/// renaming rule
///
///************************************

%rename("__%(utitle)s__", %$isfunction, notregexmatch$name="GainValue$") "";
%rename("%(utitle)s", %$isfunction, regexmatch$name="GainValue$") "";
%rename("%(utitle)s", %$ismember, %$isfunction) "";
%rename("%(utitle)s", %$ismember, %$isvariable) "";
%rename("%s", %$ismember, %$isvariable, %$isstatic, %$hasvalue) "";
%rename("%s", %$isenum) "";
%rename("%s", %$isconstructor) "";
%rename(__forward__)              CNTK::Function::Forward;
%rename(__backward__)             CNTK::Function::Backward;
%rename(__train_minibatch__)  CNTK::Trainer::TrainMinibatch;
%rename(__train_minibatchdata__)  CNTK::Trainer::TrainMinibatch(const std::unordered_map<Variable, MinibatchData>&, const DeviceDescriptor& = DeviceDescriptor::UseDefaultDevice());
%rename(__train_minibatchdata__) CNTK::Trainer::TrainMinibatch(const std::unordered_map<Variable, MinibatchData>&, std::unordered_map<Variable, ValuePtr>&, const DeviceDescriptor& = DeviceDescriptor::UseDefaultDevice());
%rename(__test_minibatchdata__) CNTK::Trainer::TestMinibatch(const std::unordered_map<Variable, MinibatchData>&, const DeviceDescriptor& = DeviceDescriptor::UseDefaultDevice());
%rename(__create__)               CNTK::Value::Create;
%rename(__dynamic_axes__)         CNTK::Variable::DynamicAxes;
// %rename(__times_transpose__)          CNTK::TransposeTimes;
%rename(l1_regularization_weight) CNTK::AdditionalLearningOptions::l1RegularizationWeight;
%rename(l2_regularization_weight) CNTK::AdditionalLearningOptions::l2RegularizationWeight;

%typecheck(1000) CNTK::NDShape const &, CNTK::NDShape {
    // '1000' is the typecheck precedence code. It means: check after basic
    // types, but before arrays. See: http://www.swig.org/Doc3.0/Typemaps.html#Typemaps_overloading
  $1 = NIL_P(rb_check_array_type($input)) ? 0 : 1;
}

%typemap(in) const CNTK::NDShape& (CNTK::NDShape tmp) {
  VALUE arry = rb_check_array_type($input);
  if(NIL_P(arry)) {
    rb_raise(rb_eArgError, "Array expected"); SWIG_fail;
  }else{
    size_t rank = RARRAY_LEN(arry);
    std::vector<size_t> dimensions(rank);
    for (int i=0; i < rank; i++) {
	VALUE elt = RARRAY_AREF(arry, i);
        dimensions[rank-i-1] = NUM2SIZET(elt);
    }
    tmp = CNTK::NDShape(dimensions);
    $1 = &tmp;
  }
}

%ignore CNTK::NDShape::operator[];

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

%extend std::unordered_map< CNTK::Variable, CNTK::ValuePtr > {
  int __set_nullptr__(const CNTK::Variable &key) {
    (*$self)[key] = CNTK::ValuePtr(nullptr);
  }
};

//
// In the future, we would just swtich to %include "CNTKLibrary.h". 
//  
namespace CNTK {

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
    CNTK::DeviceKind Type();
    std::wstring AsString();

    %extend{

      %newobject CPUDevice;
      static CNTK::DeviceDescriptor* CPUDevice(){
        return new CNTK::DeviceDescriptor(CNTK::DeviceDescriptor::CPUDevice());
      }

      %newobject GPUDevice;
      static CNTK::DeviceDescriptor* GPUDevice(unsigned int deviceId){
        return new CNTK::DeviceDescriptor(CNTK::DeviceDescriptor::GPUDevice(deviceId));
      }

      %newobject UseDefaultDevice;
      static CNTK::DeviceDescriptor* UseDefaultDevice(){
        return new CNTK::DeviceDescriptor(CNTK::DeviceDescriptor::UseDefaultDevice());
      }

      bool __eq__(const DeviceDescriptor& other){
        return (*$self) == other;
      }

    }

    static const std::vector<DeviceDescriptor>& AllDevices();

  };

  // dont change the order.
  %include "ndshape.i"
  %include "ndarrayview.i"
  %include "dictionary.i"
  %include "variable.i"
  %include "functions.i"
  %include "trainer.i"
  %include "io.i"
  %include "dist.i"
  %include "session.i"

  void ComputeInputPerDimMeansAndInvStdDevs(const MinibatchSourcePtr& minibatchSource,
                                            std::unordered_map<StreamInformation, 
                                            std::pair<NDArrayViewPtr, NDArrayViewPtr>>& computedMeanAndVariances,
                                            const DeviceDescriptor& device = DeviceDescriptor::CPUDevice());
  void SetMaxNumCPUThreads(size_t numCPUThreads);
  size_t GetMaxNumCPUThreads();


};
