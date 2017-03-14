require "test/unit"
require "cntk"
require "numo/narray"
# require "nmatrix"

class TestCNTK < Test::Unit::TestCase
  include CNTK
  include CNTK::Ops
  include Numo

  def test_highest_precision_type
    assert_equal( Numo::DFloat,
                  Ops::highest_precision_type( Numo::SFloat[1], 
                                               Numo::DFloat[1],
                                               constant(Numo::DFloat[1]) ) )
  end

  def test_input_variable
    x_ = SFloat[*(0..23).to_a].reshape(4,1,3,2)
    x  = input_variable([3,2])#, dynamic_axes: [Axis.default_batch_axis(), Axis.default_dynamic_axis] )
    f = combine([x])
    assert_equal(f.output.dynamic_axes,
                 [Axis.default_batch_axis(), Axis.default_dynamic_axis])
    assert_equal(x.shape,
                 f.output.shape)
    val = f.eval({x => x_}, remove_dynamic_axes:true)
    assert_equal(x_.shape,
                 val.shape)
    assert_equal(x_,
                 val.to_narray)
  end

  def test_backward
    x_ = SFloat[4]
    x = input_variable([1], needs_gradient: true)
    f = sqrt(x)
    state = f.forward({x => x_}, f.outputs, keep_for_backward: f.outputs)
    assert_equal(SFloat[0.25],
                 f.backward(state[0], {f.output => SFloat[1]}, [x]).values[0].to_narray)
  end

  def test_constant
    v = input_variable([3])
    f = 2 * v
    r = f.forward( { v => Numo::SFloat[1,2,3] }, f.outputs )
    assert_equal(Numo::SFloat[2,4,6],
                 r[1].values[0].to_narray)

    f = v * 2
    r = f.forward( { v => Numo::SFloat[1,2,3] }, f.outputs )
    assert_equal(Numo::SFloat[2,4,6],
                 r[1].values[0].to_narray)
  end

  def test_constant_2
    assert_equal( [2,3], 
                  constant( [[1,2,3], [4,5,6]] ).shape.to_a )
  end

  def test_transpose
    x_ = SFloat[[0,1], [2,3], [4,5]]
    x  = input_variable(x_.shape)
    assert_equal( x_.transpose,
                  transpose(x).eval({x=>x_}).to_narray)
  end

  def test_cross_entropy
    cross_entropy_with_softmax([1, 2, 3, 4], [0.35, 0.15, 0.05, 0.45]).eval.to_narray
  end

  # FIXME
  def test_lambda_rank
    output = input_variable([1])
    gain   = input_variable([1])
    group  = input_variable([1])
    lambda_rank(output, gain, group)
  end

  def test_classification_error
    x = input_variable([1])
    y = input_variable([1])
    f = classification_error(x, y)
    f.eval({ x => Numo::SFloat[1], y => Numo::SFloat[0] }).to_narray
  end

  def test_convolution
    img = SFloat[*(0..24).to_a].reshape(1,5,5)
    x = input_variable(img.shape)
    filter = SFloat[2,-1,-1,2].reshape(1,2,2)
    kernel = constant(filter)
    f = convolution(kernel: kernel, input: x, padding: [false])
    assert_equal(Numo::DFloat[[[6, 8, 10, 12], 
                               [16, 18, 20, 22], 
                               [26, 28, 30, 32], 
                               [36, 38, 40, 42]]],
                 f.eval({x => img}).to_narray.reshape(1,4,4) )
  end

  def test_roipooling
    x = input_variable([1,1,3,3])
    x_ = SFloat[[1,2,3],
                [4,5,6],
                [7,8,9]].reshape(1,1,3,3)
    rois = input_variable([1,1,4])
    rois_ = SFloat[[1/3r, 1/3r, 2/3r, 2/3r]].reshape(1,1,4)
    assert_equal( SFloat[[5,6,6],
                         [8,9,9],
                         [8,9,9]],
                  roipooling(x, rois, [3,3]).eval({x => x_, rois => rois_}).to_narray.reshape(3,3) )
  end

  def test_pooling
    x_ = SFloat[*(0..15).to_a].reshape(4,4)
    x  = input_variable(x_.shape)
    assert_equal(SFloat[[ 5,  7],
                        [13, 15]],
                 pooling(x, :max, [2,2], strides: [2,2]).eval({x => x_}).to_narray)
    assert_equal(SFloat[[ 2.5,  4.5],
                        [10.5, 12.5]],
                 pooling(x, :average, [2,2], strides: [2,2]).eval({x => x_}).to_narray)
  end

  def test_unpooling
    x_ = SFloat[*(0..15).to_a].reshape(4,4)
    x  = input_variable(x_.shape)
    operand = pooling(x, :max, [2,2], strides: [2,2])
    assert_equal(SFloat[[0, 0, 0, 0],
                        [0, 5, 0, 7],
                        [0, 0, 0, 0],
                        [0, 13, 0, 15]],
                 unpooling(operand, x, :max, [2,2], strides: [2,2]).eval({x => x_}).to_narray)
  end

  def test_batch_normalization
    x_ = SFloat[[1, 1, 2, 3],
                [0, 0, 0, 0],
                [3, 3, 4, 4]]
    x_ = x_.reshape(x_.size,1,1)
    x  = input_variable([1])
    mean     = 1
    variance = 2
    scale    = 3
    bias     = 4
    epsilon  = 0.00001
    expected_x_ = (x_ - mean) / Math::sqrt(variance + epsilon) * scale + bias
    assert_equal( expected_x_,
                  batch_normalization(x, scale: 3, bias: 4, mean: 1, variance: 2).eval({x => x_}).to_narray )
  end


  def test_log_add_exp
    x_ = SFloat[0,1,2]
    x  = input_variable(x_.shape)
    y  = input_variable(x_.shape)
    ret = log_add_exp(x, y).eval({x => NMath::log(1+x_), y => NMath.log(1+x_*x_)}).to_narray
    assert_equal( SFloat[2,4,8], NMath::exp(ret) )
  end

  def test_times_2
    x_ = SFloat[*(0..7).to_a].reshape(2,2,2)
    x  = input_variable(x_.shape)
    y  = input_variable(x_.shape)
    assert_equal( x_.reshape(2,4).dot(x_.reshape(4,2)),
                  times(x,y,1).eval({x => x_, y => x_}).to_narray )
    assert_equal( ( x_.reshape(4,2).dot x_.reshape(2,4) ).reshape(2,2,2,2),
                  times(x,y,2).eval({x => x_, y => x_}).to_narray )
  end

  # FIXME when right val is vector.
  def test_transpose_times
    x_ = SFloat[[1,2],
                [3,4]]
    assert_equal(x_.dot(x_.transpose),
                 times_transpose(x_, x_).eval.to_narray)
  end

  def test_clip
    assert_equal( SFloat[2,2.1,3,4],
                  clip([1, 2.1, 3.0, 4.1], 2, 4).eval.to_narray )
  end

  def test_element_select
    assert_equal( SFloat[1, 10, 200, 1000, 10000],
                  element_select([-10, -1, 0, 0.3, 100],
                                 [1, 10, 100, 1000, 10000],
                                 [ 2, 20, 200, 2000, 20000]).eval().to_narray )
  end

  def test_future_value
    x_ = SFloat[*(0..23).to_a].reshape(1,4,3,2)
    x  = input_variable([3,2])
    assert_equal(SFloat[[[[6, 7],
                          [8, 9],
                          [10, 11]],
                         [[12, 13],
                          [14, 15],
                          [16, 17]],
                         [[18, 19],
                          [20, 21],
                          [22, 23]],
                         [[0, 0],
                          [0, 0],
                          [0, 0]]]],
                 future_value(x).eval({x => x_}).to_narray )
  end

  def test_past_value
    x_ = SFloat[*(0..23).to_a].reshape(1,4,3,2)
    x  = input_variable([3,2])
    assert_equal(SFloat[[[[0, 0],
                          [0, 0],
                          [0, 0]],
                         [[0, 1],
                          [2, 3],
                          [4, 5]],
                         [[6, 7],
                          [8, 9],
                          [10, 11]],
                         [[12, 13],
                          [14, 15],
                          [16, 17]]]],
                 past_value(x).eval({x => x_}).to_narray)
  end

  def test_reshape
    x_ = SFloat[[0,1],[2,3],[4,5]]
    x  = input_variable([3,2])
    assert_equal(x_.reshape(2,3),
                 reshape(x, [2,3]).eval({x => x_}).to_narray)
  end

  def test_slice
    x_ = SFloat[[1,2,3],[4,5,6]]
    x  = input_variable(x_.shape)
    assert_equal(SFloat[[4,5,6]],
                 slice(x, 0, 1, 2).eval({x => x_}).to_narray)
    assert_equal(SFloat[[1],[4]],
                 slice(x, 1, 0, 1).eval({x => x_}).to_narray)
  end

  def test_splice
    x_ = SFloat[[[1,2],
                 [4,5]]]
    y_ = SFloat[[[10,20],
                 [30,40],
                 [50,60]]]
    assert_equal( SFloat[[[1, 2],
                          [4, 5],
                          [10, 20],
                          [30, 40],
                          [50, 60]]],
                  splice([x_, y_], 1).eval.to_narray)
  end

  def test_reduce_sum
    x_ = SFloat[*(0..15).to_a].reshape(2,2,2,2)
    x  = input_variable([2,2])
    reduce_sum(x).eval({x => x_}).to_narray
  end

  def test_reduce_log_sum_exp
    x_ = SFloat[*(0..5).to_a].reshape(1,3,2)
    x  = input_variable([3,2])
    assert_equal( SFloat[[5.4561934]],
                  reduce_log_sum_exp(x).eval({x => x_}).to_narray )
  end

  def test_reduce_mean
    x_ = SFloat[[5, 20],[30, 40],[55, 60]]
    x  = input_variable(x_.shape)
    assert_equal(SFloat[[30,40]],
                 reduce_mean(x,0).eval({x => x_}).to_narray)
    assert_equal(SFloat[[12.5],
                        [35.0],
                        [57.5]],
                 reduce_mean(x,1).eval({x => x_}).to_narray)
  end

  def test_random_sample
    eye_ = SFloat.eye(4)
    ret = random_sample(SFloat[0.1,0.1,0.1,0.7], 10, true).eval({})
    times( ret, eye_ ).eval.to_narray
  end

  # FIXME
  def test_edit_distance_error
    x = input_variable([2])
    y = input_variable([2])
    edit_distance_error(x, y, 0, 1, 1, true, [1])
#    f.eval({ x =>  Numo::SFloat[[1, 3], [2, 0]], y => Numo::SFloat[[2, 0], [2, 0]] }).shape
  end

end

#"__negate__", "__sigmoid__", "__tanh__", "__cos__", "__re_lu__", "__exp__", "__log__", "__square__", "__sqrt__", "__round__", "__floor__", "__ceil__", "__reciprocal__", "__softmax__", "__hardmax__", "__transpose_axes__", "__transpose__", "__slice__", "__random_sample__", "__random_sample_inclusion_frequency__", "__dropout__", "__reshape__", "__plus__", "__minus__", "__log_add_exp__", "abs", "__element_times__", "__element_divide__", "__equal__", "__not_equal__", "__less__", "__less_equal__", "__greater__", "__greater_equal__", "__transpose_times__", "__cosine_distance__", "__binary_cross_entropy__", "__weighted_binary_cross_entropy__", "__squared_error__", "__cross_entropy_with_softmax__", "__classification_error__", "__lambda_rank__", "__ndcgat1__", "__past_value__", "__future_value__", "__reduce_sum__", "__reduce_log_sum__", "__reduce_mean__", "__reduce_max__", "__reduce_min__", "__per_dim_mean_variance_normalize__", "__convolution__", "__roipooling__", "__pooling__", "__unpooling__", "__batch_normalization__", "__optimized_rnnstack__", "__clip__", "__element_select__", "__splice__", "__combine__", "__alias__", "__as_block__", "__is_first__", "__is_last__", "__first__", "__last__", "__where__", "__gather__", "__scatter__", "__broadcast_as__"
