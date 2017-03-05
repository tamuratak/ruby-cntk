module CNTK
class Axis
class << self
  def from_num(n)
    case n
    when Axis
      n
    when Numeric
      new(-n-1)
    else
      raise ArgumentError
    end
  end
end
end
end
