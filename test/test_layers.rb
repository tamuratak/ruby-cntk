require "test/unit"
require "cntk"
require "numo/narray"

class TestCNTK < Test::Unit::TestCase
  include CNTK
  include CNTK::Ops

  def test_dense
    x = input_variable([3])
    f = Layers.dense([2])
    f.(x)
  end

end
