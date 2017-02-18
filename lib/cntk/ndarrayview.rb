module CNTK
  class NDArrayView

    def self.create(a)
      if a.respond_to?(:shape) and a.respond_to?(:row_major?)
        if a.row_major?
          # NDArrayView is column-major.
          # So we must transpose a.
          ta = a.transpose
        end
        case a
        when Numo::DFloat
          dtype = DataType_Double
        when Numo::SFloat
          dtype = DataType_Float
        else
          raise "not implemented"
        end
        return self.new(dtype, a.shape, ta.flatten.to_a,
                        CNTK::DeviceDescriptor.default_device(), false)
      else
        raise "not implemented"
      end
    end

    def to_narray
      ret = Numo::DFloat[*to_vec()]
      # NDArrayView is column-major and NArray is row-major.
      # So we must reverse shape and transpose it.
      ret = ret.reshape(*shape().reverse)
      return ret.transpose
    end

  end
end
