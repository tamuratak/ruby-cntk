
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
    DictionaryValue(const std::vector<RubyCNTK::DictionaryValue>&);
    DictionaryValue(const Axis&);
    DictionaryValue(const std::wstring&);
    DictionaryValue(const RubyCNTK::Dictionary&);
    DictionaryValue(const RubyCNTK::DictionaryValue&);

    bool HasValue();
    enum Type ValueType();

    void Save(const std::wstring& filename);
    static DictionaryValue Load(const std::wstring& filename);
    static const char* TypeName(Type type);

    %extend{
      /*
      DictionaryValue(const std::vector<RubyCNTK::DictionaryValue>& value){
        return new ::CNTK::DictionaryValue( (std::vector<::CNTK::DictionaryValue>) value);
      }
      */
      bool __eq__(const RubyCNTK::DictionaryValue& other) {
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
        return $self->Value<RubyCNTK::Axis>();
      }

      std::wstring& Value_wstring__() {
        return $self->Value<std::wstring>();
      }

      std::vector<RubyCNTK::DictionaryValue>& Value_vec_dict_value__() {
        return $self->Value<std::vector<RubyCNTK::DictionaryValue> >();
      }

      RubyCNTK::Dictionary& Value_dict__() {
        return $self->Value<RubyCNTK::Dictionary>();
      }

      RubyCNTK::NDArrayView& Value_ndarrayview__() {
        return $self->Value<RubyCNTK::NDArrayView>();
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
      RubyCNTK::DictionaryValue __getitem__(const std::wstring& key) {
        return (*$self)[key];
      }

      void __setitem__(const std::wstring& key, const RubyCNTK::DictionaryValue& v) {
        (*$self)[key] = v;
      }

      bool __eq__(const Dictionary& other) {
        return (*$self) == other;
      }
    }
  };
