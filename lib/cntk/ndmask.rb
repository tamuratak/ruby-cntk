require "numo/narray"
module CNTK
  class NDMask
    def to_narray
      Numo::Int8[*to_vec()].reshape(*shape())
    end
  end
end
