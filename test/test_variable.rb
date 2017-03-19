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

  def test_times_3
    x_ = constant( SFloat[1,2,3] ).reshape( [1,3] )
    y_ = constant( SFloat[1,2,3] ).reshape( [3,1] )
    b_ = constant( SFloat[1].reshape(1,1) )
    ret = x_.dot(y_)
    ret.outputs[0].shape
    ret2 = ret.outputs[0] + b_
    ret2.outputs[0].shape
  end

end
