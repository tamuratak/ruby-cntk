require "test/unit"
require "cntk"
require "numo/narray"

class TestCNTKReader < Test::Unit::TestCase
  include CNTK
  include CNTK::Ops
  include Numo

  def test_value_class
    val = Value.create(input_variable([]), [SFloat[5,6,7], SFloat[8]],
                           [true, false], DeviceDescriptor.use_default_device)
    assert_equal(Int8[[2,1,1],[1,0,0]],
                 val.mask.to_narray)
  end

end
