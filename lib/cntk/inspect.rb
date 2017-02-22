module CNTK
  module InspectUtil
    def inspect_methods_p(mthds)
      s = mthds.map{|mth| "#{mth}=" + send(mth).inspect }.join(", ")
    end

    def inspect_methods(mthds)
      s = inspect_methods_p(mthds)
      "#<#{self.class} #{s}>"
    end

  end

  class StdUMapStreamInfoMinibatchData
    include InspectUtil
    def inspect
      s = "{" + map{|k, v| k.inspect + " => " + v.inspect }.join(", ") + "}"
      "#<#{self.class}: #{s}>"
    end
  end

  class NDShape
    def inspect
      to_a.inspect
    end
  end

  class Value
    include InspectUtil
    def inspect
      inspect_methods([:shape])
    end
  end

  class StreamInformation
    include InspectUtil

    def inspect
      inspect_methods([:name, :id])
    end
  end

  class MinibatchData
    include InspectUtil

    def inspect
      inspect_methods([:data, :number_of_samples])
    end
  end

end
