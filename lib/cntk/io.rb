module CNTK

  def self.create_composite_minibatch_source(dict)
    if dict.respond_to?(:to_hash)
      h = {}
      dict.to_hash.each_pair{|k, v|
        k = k.to_s if k.is_a?(Symbol)
        h[k] = v
      }
      des = h["deserializers"]
      unless des.respond_to?(:to_ary)
        h["deserializers"] = [des]
      end
      dict = Dictionary.create(h)
    end
    CNTK.__create_composite_minibatch_source__(dict)
  end

  class MinibatchSource

    # @param minibatch_size_in_samples [Integer]
    # @param device                    [DeviceDescriptor]
    # @param num_data_partitions       [Integer]
    # @param partition_index           [Integer]
    # @return [MinibatchData]
    def next_minibatch(minibatch_size_in_samples, device: DeviceDescriptor.use_default_device,
                       num_data_partitions: 1, partition_index: 0)
      mb = get_next_minibatch(0, minibatch_size_in_samples, num_data_partitions, partition_index, device)
    end

  end

  # std::unordered_map<StreamInfo, MinibatchData>
  class MinibatchTable 
    alias __get__ :[]
    def [](key)
      if key.respond_to?(:to_str)
        key = key.to_str
        a = self.keys.find_all{|k| k.name == key }
        if a.size > 1
          raise "The number of input data having the name is not 1."
        end
        __get__(a[0])
      else
        __get__(key)
      end
    end
  end

end
