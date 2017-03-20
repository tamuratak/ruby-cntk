module CNTK
  class NDArrayView

    def self.create(a)
      if a.respond_to?(:shape)
        case a
        when NDArrayView
          return a
        when Numo::DFloat
          dtype = DataType_Double
        when Numo::SFloat
          dtype = DataType_Float
        else
          raise ArgumentError, "Numo::NArray or NDArrayView expected"
        end
        return self.new(dtype, a.shape, a.flatten.to_a,
                        CNTK::DeviceDescriptor.use_default_device(), false)
      else
        raise ArgumentError, "not responds to :shape"
      end
    end

    def to_narray
      case get_data_type
      when DataType_Float
        klass = Numo::SFloat
      when DataType_Double
        klass = Numo::DFloat
      else
        raise "unknown data type"
      end
      ret = klass[*to_vec()]
      ret = ret.reshape(*shape().to_a)
      return ret
    end

  end
end
