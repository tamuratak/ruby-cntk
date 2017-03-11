module CNTK
class Axis
class << self
  def from_num(n)
    case n
    when nil
      all_static_axes
    when Numeric
      new(-n-1)
    when Axis
      new(-1-n.static_axis_index)
    else
      raise ArgumentError
    end
  end
end
end
end
