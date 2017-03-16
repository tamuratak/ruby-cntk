module CNTK
  class NDArrayView

    def self.create(a)
      if a.respond_to?(:shape) and a.respond_to?(:row_major?)
        if a.row_major?
          # NDArrayView is column-major.
          # So we must transpose a.
          ta = a #a.transpose
        end
        case a
        when NDArrayView
          return a
        when Numo::DFloat
          dtype = DataType_Double
        when Numo::SFloat
          dtype = DataType_Float
        else
          raise "not implemented"
        end
        return self.new(dtype, a.shape.to_a, ta.flatten.to_a,
                        CNTK::DeviceDescriptor.default_device(), false)
      else
        raise "not implemented"
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
      # NDArrayView is column-major and NArray is row-major.
      # So we must reverse shape and transpose it.
      ret = ret.reshape(*shape().to_a)
      return ret #.transpose
    end

  end
end
