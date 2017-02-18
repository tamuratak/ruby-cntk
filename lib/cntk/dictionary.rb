module CNTK

  class DictionaryValue
    
    def self.create(val)
      case val
      when Hash
        new(Dictionary.create(val))
      else
        new(val)
      end
    end

    def value
      case value_type
      when Type_Bool
        value_bool__
      when Type_Int
        value_int__
      when Type_SizeT
        value_size_t__
      when Type_Float
        value_float__
      when Type_Double
        value_double__
      when Type_String
        value_string__
      when Type_NDShape
        value_ndshape__
      when Type_Axis
        value_axis__
      when Type_Vector
        value_vec_dict_value__
      when Type_Dictionary
        value_dictionary__
      when Type_NDArrayView
        value_ndarrayview__
      else
        raise "unknown type"
      end
    end

  end

  class Dictionary
    def self.create(h)
      dict = new()
      h.each_pair{|k, v|
        dict[k] = DictionaryValue.create(v)
      }
      return dict
    end
  end

end
