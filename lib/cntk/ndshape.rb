module CNTK
  class NDShape
    def to_ary
      dimensions.reverse
    end

    def to_a
      to_ary
    end

    def reverse
      dimensions
    end
  end
end
