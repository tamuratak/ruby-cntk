module CNTK

  class Function

    def coerce(other)
      if other.is_a?(Numeric)
        [Constant::scalar(output.get_data_type, other), self]
      else
        
      end
    end

    def call(func)
      if func.outputs.length == 1
        return replace_placeholders({placeholders[0] => func.output})
      else
        raise "the outputs of given Funtion object must has 1 length."
      end
    end

    # forward function composition self(func(...))
    def >>(func)
      func.call(self)
    end

    def <<(func)
      call(func)
    end

    def forward(argsmap, outmap: nil, keep_for_backward: false, device: DeviceDescriptor.use_default_device(), remove_dynamic_axes: true)
      input = convert_to_value(argsmap)
      out = StdUMapVariableValue.new()
      outputs().each{|out_var| 
        # By setting nullptr, Forward function implemented in C++ will allocate Value object with required storage.
        out.__set_nullptr__(out_var)
      }
      b = __forward__(input, out)
      # FIXME. we will remove this line.
      out = remove_dynamic_axes(out) if remove_dynamic_axes
      return [b, out]
    end

    def eval(argsmap=nil, device: DeviceDescriptor.use_default_device(), remove_dynamic_axes: true)
      argsmap = {} if argsmap == nil
      _, outmap = forward(argsmap, device: device, remove_dynamic_axes: remove_dynamic_axes)
      if outmap.size > 1
        outmap
      else
        outmap.values[0]
      end
    end

    def convert_to_value(h)
      input = {}
      h.each_pair{|k,val|
        if val.respond_to?(:row_major?)
          input[k] = Value.create(val)
        else
          input[k] = val
        end
      }
      return input
    end

    def remove_dynamic_axes(out)
      out1 = {}
      out.each{|o,ov|
        sz = o.dynamic_axes.size
        if sz > 0 and sz < ov.shape.rank and ov.shape.to_a[0..1] == [1,1]
          out1[o] = ov.reshape( ov.shape.to_a[sz..-1] )
        else
          out1[o] = ov
        end
      }
      return out1
    end

    private :__forward__
  end
end
