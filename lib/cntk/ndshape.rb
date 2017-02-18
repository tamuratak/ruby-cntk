module CNTK
  class NDShape
    alias :to_ary :dimensions
    alias :to_a   :dimensions
    
    def reverse
      to_ary.reverse
    end
  end
end
