require "test/unit"
require "cntk"
require "numo/narray"
# require "nmatrix"

class TestCNTK < Test::Unit::TestCase
  include CNTK
  include CNTK::Ops

  def test_f
    v = NDArrayView.new(DataType_Float, [2], [1.9, 1.2], DeviceDescriptor.default_device(), true)
    v = Value.new(v)
    x = input_variable([2])
    f = __sin__(x)
    out = StdUMapVariableValue.new()
    vo = NDArrayView.new(DataType_Float, [2,1,1], [1.0,1.0], DeviceDescriptor.default_device(), true)
    f.output.shape
    f.output.shape.rank
    valout = Value.new(vo)
    out[f.output] = valout
    f.forward({x => v}, out)
    f.forward({x => v})
    NDArrayView.new(1.9)
  end

  def test_ndshape
    NDShape.new([1,1,1]).to_ary
    assert_equal([1,2,3], NDShape.new([1,2,3]).to_ary)
  end

  def test_times
    v0 = NDArrayView.new(DataType_Float, [2], [1.0,0.0], DeviceDescriptor.default_device(), true)
    v = input_variable([2])
    m0 = NDArrayView.new(DataType_Float, [3,2], [1,2,3,4,5,6], DeviceDescriptor.default_device(), true)
    m = input_variable([3,2])
    f = __times__(m,v)
    r = f.forward({v => Value.new(v0), m => Value.new(m0)})
    assert_equal([1.0,2.0,3.0], r[0].values[0].data.to_vec)

    r = f.forward( { v => Numo::SFloat[1,0], 
                     m => Numo::SFloat[[1,2],[3,4],[5,6]] })
    assert_equal([1.0,3.0,5.0], r[0].values[0].data.to_vec)

    m = input_variable([3,1])
    v = input_variable([1,2])
    f = __times__(m,v)
    r = f.forward( { v => Numo::SFloat[1,2].reshape(1,2),
                     m => Numo::SFloat[1,3,4].reshape(3,1) } )
    assert_equal(Numo::SFloat[[1,2],[3,6],[4,8]],
                 r[0].values[0].to_narray )
  end

  def test_times_edge_cases
    v = input_variable([3])
    m = input_variable([3])
    f = __times__(v,m)
    r = f.forward( { v => Numo::SFloat[1,0,0].reshape(3),
                     m => Numo::SFloat[1,2,3].reshape(3) } )
    assert_equal(Numo::SFloat[[1, 2, 3], 
                              [0, 0, 0], 
                              [0, 0, 0]],
                 r[0].values[0].to_narray )
  end

  def test_constant
    v = input_variable([3])
    f = 2 * v
    r = f.forward( { v => Numo::SFloat[1,2,3] } )
    assert_equal(Numo::SFloat[2,4,6],
                 r[0].values[0].to_narray)

    f = v * 2
    r = f.forward( { v => Numo::SFloat[1,2,3] } )
    assert_equal(Numo::SFloat[2,4,6],
                 r[0].values[0].to_narray)
  end

  def test_parameter
    parameter(init: 2)
    parameter(init: uniform_initializer(1) )
  end

  def test_scalar
    s = input_variable([])
    v = input_variable([1])
    __times__(v,s)
  end

  def test_function_call
    v0 = NDArrayView.new(DataType_Double, [1], [1.9], DeviceDescriptor.default_device(), true)
    v = input_variable([1], dtype: DataType_Double)
    f1 = sin(v)
    x = placeholder_variable()
    f2 = sin(x)
    f3 = f2.(f1)
    r = f3.forward({v => Value.new(v0)})
    assert_equal([Math::sin(Math::sin(1.9))],
                 r[0].values[0].data.to_vec)
  end

  def test_function_forward
    v = input_variable([1])
    f = CNTK.__sin__(v)
    r = f.forward({ v => Numo::SFloat[Math::PI/2] })
    assert_equal([1.0], r[0].values[0].data.to_vec )
  end

  def test_func_shift_op
    v0 = NDArrayView.new(DataType_Double, [1], [1.9], DeviceDescriptor.default_device(), true)
    v = input_variable([1], dtype: DataType_Double)
    f1 = sin(v)
    x = placeholder_variable()
    f2 = sin(x)
    f3 = f1 >> f2
    r = f3.forward({v => Value.new(v0)})
    assert_equal([Math::sin(Math::sin(1.9))],
                 r[0].values[0].data.to_vec)
  end

  def test_narray
    v = NDArrayView.new(DataType_Float, [3,2], [1,2,3,4,5,6], DeviceDescriptor.default_device(), true)
    assert_equal([[1.0, 4.0], 
                  [2.0, 5.0], 
                  [3.0, 6.0]], 
                 v.to_narray.to_a)

    v = NDArrayView.new(DataType_Float, [2,2,3], (1..12).to_a, DeviceDescriptor.default_device(), true)
    assert_equal([ [[1.0, 5.0, 9.0],  
                    [3.0, 7.0, 11.0]], 
                   [[2.0, 6.0, 10.0], 
                    [4.0, 8.0, 12.0]] ],
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
    assert_equal(m, Value.create(m).to_narray)
  end

  def test_dictionary
    d = Dictionary.create({"a" => 1})
    assert_equal(1, d["a"].value)
  end

end
