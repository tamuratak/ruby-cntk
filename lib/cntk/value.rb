module CNTK
  class Value

    def self.create(a)
      new(NDArrayView.create(a))
    end

    def to_narray
      data().to_narray
    end

    def reshape(a)
      na = to_narray().reshape(*a)
      self.class.create(na)
    end

  end
end
