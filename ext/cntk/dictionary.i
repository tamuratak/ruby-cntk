
  class DictionaryValue
  {
  public:
    enum class Type : unsigned int
    {
        None,
        Bool,
        Int,
        SizeT,
        Float,
        Double,
        String,
        NDShape,
        Axis,
        Vector,
        Dictionary,
        NDArrayView,
    };

    DictionaryValue();
    ~DictionaryValue();

    DictionaryValue(bool);
    DictionaryValue(size_t);
    DictionaryValue(double);
    DictionaryValue(const std::vector<CNTK::DictionaryValue>&);
    DictionaryValue(const Axis&);
    DictionaryValue(const std::wstring&);
    DictionaryValue(const CNTK::Dictionary&);
    DictionaryValue(const CNTK::DictionaryValue&);

    bool HasValue();
    enum Type ValueType();

    void Save(const std::wstring& filename);
    static DictionaryValue Load(const std::wstring& filename);
    static const char* TypeName(Type type);

    %extend{
      /*
      DictionaryValue(const std::vector<CNTK::DictionaryValue>& value){
        return new ::CNTK::DictionaryValue( (std::vector<::CNTK::DictionaryValue>) value);
      }
      */
      bool __eq__(const CNTK::DictionaryValue& other) {
        return (*$self) == other;
      }

      bool Value_bool__() {
        return $self->Value<bool>();
      }

      int Value_int__() {
        return $self->Value<int>();
      }

      size_t Value_size_t__() {
        return $self->Value<size_t>();
      }

      float Value_float__() {
        return $self->Value<float>();
      }

      double Value_double__() {
        return $self->Value<double>();
      }

      Axis& Value_axis__() {
        return $self->Value<CNTK::Axis>();
      }

      std::wstring& Value_wstring__() {
        return $self->Value<std::wstring>();
      }

      std::vector<CNTK::DictionaryValue>& Value_vec_dict_value__() {
        return $self->Value<std::vector<CNTK::DictionaryValue> >();
      }

      CNTK::Dictionary& Value_dict__() {
        return $self->Value<CNTK::Dictionary>();
      }

      CNTK::NDArrayView& Value_ndarrayview__() {
        return $self->Value<CNTK::NDArrayView>();
      }

    }
  };

  class Dictionary {
  public:
    Dictionary();
    ~Dictionary();

    Dictionary(const Dictionary&);
    bool Contains(const std::wstring& key);
    void Add(const Dictionary& other);
    size_t Size();
    void Save(const std::wstring& filename);
    static Dictionary Load(const std::wstring& filename);

    %extend{
      CNTK::DictionaryValue __getitem__(const std::wstring& key) {
        return (*$self)[key];
      }

      void __setitem__(const std::wstring& key, const CNTK::DictionaryValue& v) {
        (*$self)[key] = v;
      }

      bool __eq__(const Dictionary& other) {
        return (*$self) == other;
      }
    }
  };
