module CNTK
  class Value

    def self.create(variable, data, seq_starts=[], device=DeviceDescriptor.use_default_device, read_only=false)
      if variable.dynamic_axes.size == 0
        ndav = NDArrayView.new(data, DeviceDescriptor.cpudevice)
        new(ndav)
      else
        ndav = data.map{|a| NDArrayView.create(a) }
        __create__(variable.shape, ndav, seq_starts, device, read_only)
      end
    end

    def to_narray
      data().to_narray
    end

    def reshape(a)
      na = to_narray().reshape(*a)
      self.class.new(NDArrayView.create(na))
    end

  end
end
