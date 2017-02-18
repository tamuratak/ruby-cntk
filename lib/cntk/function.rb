module CNTK
  class Function

    def coerce(other)
      if other.is_a?(Numeric)
        [Constant::scalar(output.get_data_type, other), self]
      else
        
      end
    end

    def call(args)
      if args.outputs.length == 1
        return replace_placeholders({placeholders[0] => args.output})
      else
        raise "not implemented"
      end
    end

    def forward(*args)
      if args.length > 1
        return __forward__(*args)
      elsif args.length == 1
        input = convert_to_value(args[0])
        out = StdUMapVariableValue.new()
        outputs().each{|o|
          v = NDArrayView.new(o.get_data_type(),
                              required_output_shape(o),
                              required_output_buf(o),
                              CNTK::DeviceDescriptor.default_device(),
                              true)
          out[o] = Value.new(v)
        }
        b = __forward__(input, out)
        out = remove_dynamic_axes(out)
        return [out, b]
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

    #FIXME
    # we must add dynamic axes?
    def required_output_shape(ov)
      ov.shape().to_a + [1,1]
    end

    def required_output_buf(ov)
      [1.0] * ov.shape.total_size
    end

    def remove_dynamic_axes(out)
      out1 = {}
      out.each{|o,ov|
        if ov.shape.rank == o.shape.rank + 2 and ov.shape.to_a[-2..-1] == [1,1]
          out1[o] = ov.reshape( ov.shape.to_a[0..-3] )
        else
          out1[o] = ov
        end
      }
      return out1
    end

    private :__forward__
  end
end