require "test/unit"
require "cntk"
require "numo/narray"

class TestCNTK < Test::Unit::TestCase
  include CNTK
  include CNTK::Ops

  def test_ndshape
    NDShape.new([1,1,1]).to_ary
    assert_equal([3,2,1], NDShape.new([1,2,3]).to_ary)
  end

  def test_times_edge_cases
    v = input_variable([3])
    m = input_variable([3])
    f = CNTK.__times__(v,m)
    r = f.eval( { v => Numo::SFloat[1,0,0].reshape(3),
                  m => Numo::SFloat[1,2,3].reshape(3) } )
    assert_equal(Numo::SFloat[[1, 0, 0], 
                              [2, 0, 0], 
                              [3, 0, 0]],
                 r.to_narray.reshape(3,3) )
  end

  def test_parameter
    parameter(init: 2)
    parameter(init: CNTK.uniform_initializer(1) )
  end

  def test_scalar
    s = input_variable([])
    v = input_variable([1])
    CNTK.__times__(v,s)
  end



  def test_narray
    v = NDArrayView.new(DataType_Float, [2,3], [1,2,3,4,5,6], DeviceDescriptor.default_device(), true)
    assert_equal([2,3],
                 v.to_narray.shape.to_a)
    assert_equal([[1, 2, 3],
                  [4, 5, 6]],
                 v.to_narray.to_a)

    v = NDArrayView.new(DataType_Float, [3,2,2], (1..12).to_a, DeviceDescriptor.default_device(), true)
    assert_equal([3,2,2],
                 v.shape.to_a)
    assert_equal([3,2,2],
                 v.to_narray.shape.to_a)
    assert_equal([[[1.0, 2.0], [3.0, 4.0]],
                  [[5.0, 6.0], [7.0, 8.0]],
                  [[9.0, 10.0], [11.0, 12.0]]],
                 v.to_narray.to_a)

    m = Numo::SFloat[1,2,3,4,5,6].reshape(2,3)
    assert_equal( m,
                  NDArrayView.create( m ).to_narray )

    m = Numo::SFloat[1,2,3,4,5,6].reshape(2,3)
    assert_equal( m,
                  NDArrayView.create( m ).to_narray )

    m = Numo::DFloat[*(1..12).to_a].reshape(2,2,3)
    assert_equal( m,
                  NDArrayView.create( m ).to_narray )

  end

  def test_value
    m = Numo::SFloat[1,2,3,4,5,6].reshape(2,3)
    assert_equal(m, Value.new( NDArrayView.create(m) ).to_narray)
  end

  def test_dictionary
    d = Dictionary.create({"a" => 1})
    assert_equal(1, d["a"].value)
  end

end
