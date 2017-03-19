require "test/unit"
require "cntk"
require "numo/narray"

class TestCNTK < Test::Unit::TestCase
  include CNTK
  include CNTK::Ops

  def test_function_call
    v0 = NDArrayView.new(DataType_Double, [1], [1.9], DeviceDescriptor.default_device(), true)
    v = input_variable([1], dtype: DataType_Double)
    f1 = sin(v)
    x = placeholder_variable()
    f2 = sin(x)
    f3 = f2.(f1)
    r = f3.eval({v => Value.new(v0)})
    assert_equal([Math::sin(Math::sin(1.9))],
                 r.data.to_vec)
  end

  def test_function_forward
    v = input_variable([1])
    f = Ops.sin(v)
    r = f.eval({ v => Numo::SFloat[Math::PI/2] })
    assert_equal([1.0], r.data.to_vec )
  end

  def test_func_shift_op
    v = input_variable([1], dtype: DataType_Double)
    f = sin() >> sin()
    f = f.(v)
    r = f.eval({v => Numo::DFloat[1.9]})
    assert_equal([ Math::sin( Math::sin(1.9) )],
                 r.data.to_vec)
  end

  
end
