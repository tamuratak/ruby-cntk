module CNTK

  class Function

    def dot(other)
      output.dot(other)
    end

    def -@
      - output
    end

    def +(other)
      output + other
    end

    def -(other)
      output - other
    end

    def *(other)
      output * other
    end

    def /(other)
      output / other
    end

    # FIXME
    def coerce(other)
      if other.is_a?(Numeric)
        [Constant::scalar(output.get_data_type, other), self]
      else
        
      end
    end

    def call(func)
      if func.respond_to?(:output)
        val = func.output
      else
        val = func
      end
      if placeholders().length == 1
        replace_placeholders({placeholders[0] => val})
      else
        raise "the number of placeholders is not 1."
      end
    end

    # forward function composition self(func(...))
    def >>(func)
      func.call(self)
    end

    def <<(func)
      call(func)
    end

    def forward(argsmap, outputs = [], keep_for_backward: [], device: DeviceDescriptor.use_default_device(), remove_dynamic_axes: true)
      input = convert_to_value(argsmap)
      out = StdUMapVariableValue.new()
      outputs.each{|out_var|
        # By setting nullptr, Forward function implemented in C++ will allocate Value object with required storage.
        out.__set_nullptr__(out_var)
      }
      b = __forward__(input, out, device, keep_for_backward)
      # FIXME. we will remove this line.
      out = remove_dynamic_axes(out) if remove_dynamic_axes
      return [b, out]
    end

    def eval(argsmap=nil, device: DeviceDescriptor.use_default_device(), remove_dynamic_axes: true)
      argsmap = {} if argsmap == nil
      _, outmap = forward(argsmap, outputs(), device: device, remove_dynamic_axes: remove_dynamic_axes)
      if outmap.size > 1
        outmap
      else
        outmap.values[0]
      end
    end

    def backward(state, root_gradients, variables, remove_dynamic_axes: true)
      root_gradients = convert_to_value(root_gradients)
      out = StdUMapVariableValue.new()
      variables.each{|var|
        out.__set_nullptr__(var)
      }
      __backward__(state, root_gradients, out)
      out = remove_dynamic_axes(out)
    end

    def convert_to_value(h)
      ret = {}
      h.each_pair{|k,val|
        if val.respond_to?(:row_major?)
          ret[k] = Value.new(NDArrayView.create(val))
        else
          ret[k] = val
        end
      }
      return ret
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
