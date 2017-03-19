require "numo/narray"

module CNTK

  class Variable
    DataType = { DataType_Float => Numo::SFloat, DataType_Double => Numo::DFloat, DataType_Unknown => nil }

    def is_scalar
      shape().rank == 0
    end

    def dynamic_axes
      __dynamic_axes__.reverse
    end

    def reshape(shape)
      Ops.reshape(self, shape)
    end

    # follow Numo::Narray
    def dot(other)
      if shape.rank == 1 and other.shape.rank == 1
        ret = Ops.times( Ops.reshape(self,  [1,shape.total_size]),
                         Ops.reshape(other, [other.shape.total_size, 1]) )
        Ops.reshape(ret, [1])
      else
        Ops.times(self, other)
      end
    end

    def *(other)
      Ops.element_times(self, other)
    end

    def /(other)
      Ops.element_divide(self, other)
    end

    def +(other)
      Ops.plus(self, other)
    end

    def -(other)
      Ops.minus(self, other)
    end

    def coerce(other)
      if other.is_a?(Numeric)
        [CNTK::Ops.constant(other), self]
      else
        [other, self]
      end
    end

  end

  module VariableExtend
    def create_from(val, shape, dtype, device, name)
      case val
      when NDArrayView, Value, Variable, Numo::NArray
        if shape
          raise "can't accept #{val.class} and shape at the same time" 
        end
        case val
        when Variable
          new(val)
        when NDArrayView
          new(val, name)
        when Value
          new(val.data, name)
        else
          new(NDArrayView.create(val), name)
        end
      when Numeric, Dictionary
        if shape
          new(shape, dtype, val, device, name)
        else
          new([], dtype, val, device, name)
        end
      else
        raise ArgumentError, "NDArrayView, Value, Variable, Numo::NArray, Initializer, or Numeric expected"
      end
    end
  end

  class Constant # < Variable
    extend VariableExtend
    def self.create(val, shape: nil, dtype: DataType_Float, device: DeviceDescriptor::use_default_device(), name: "")
      create_from(val, shape, dtype, device, name)
    end

    def *(other)
      if is_scalar
        CNTK.__times__(other, self)
      else
        CNTK.__times__(self, other)
      end
    end

  end

  class Parameter # < Variable
    extend VariableExtend
    def self.create(shape: nil, init: 0, dtype: DataType_Float, device: DeviceDescriptor::use_default_device(), name: "")
      create_from(init, shape, dtype, device, name)
    end
  end

end
