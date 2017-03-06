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
                  OpsUtil::highest_precision_type( Numo::SFloat[1], 
                                                   Numo::DFloat[1],
                                                   constant(Numo::DFloat[1]) ) )
  end

  def test_constant
    v = input_variable([3])
    f = 2 * v
    r = f.forward( { v => Numo::SFloat[1,2,3] } )
    assert_equal(Numo::SFloat[2,4,6],
                 r[1].values[0].to_narray)

    f = v * 2
    r = f.forward( { v => Numo::SFloat[1,2,3] } )
    assert_equal(Numo::SFloat[2,4,6],
                 r[1].values[0].to_narray)
  end

  def test_constant_2
    assert_equal( [2,3], constant( [[1,2,3], [4,5,6]] ).shape )
  end

  def test_transpose
    assert_equal( Numo::DFloat[*[[0,1],[2,3],[4,5]] ].transpose(1,0),
                  transpose([[0,1],[2,3],[4,5]], 0, 1).eval.to_narray)
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
#    p classification_error(Numo::SFloat[1], Numo::SFloat[1]).eval
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
                 f.eval({x => img}).to_narray )
  end

  # FIXME
  def test_edit_distance_error
    x = input_variable([2])
    y = input_variable([2])
    f = edit_distance_error(x, y, 0, 1, 1, true, [1])
    p f.eval({ x =>  Numo::SFloat[[1, 3], [2, 0]], y => Numo::SFloat[[2, 0], [2, 0]] }).shape
  end

end

#"__negate__", "__sigmoid__", "__tanh__", "__cos__", "__re_lu__", "__exp__", "__log__", "__square__", "__sqrt__", "__round__", "__floor__", "__ceil__", "__reciprocal__", "__softmax__", "__hardmax__", "__transpose_axes__", "__transpose__", "__slice__", "__random_sample__", "__random_sample_inclusion_frequency__", "__dropout__", "__reshape__", "__plus__", "__minus__", "__log_add_exp__", "abs", "__element_times__", "__element_divide__", "__equal__", "__not_equal__", "__less__", "__less_equal__", "__greater__", "__greater_equal__", "__transpose_times__", "__cosine_distance__", "__binary_cross_entropy__", "__weighted_binary_cross_entropy__", "__squared_error__", "__cross_entropy_with_softmax__", "__classification_error__", "__lambda_rank__", "__ndcgat1__", "__past_value__", "__future_value__", "__reduce_sum__", "__reduce_log_sum__", "__reduce_mean__", "__reduce_max__", "__reduce_min__", "__per_dim_mean_variance_normalize__", "__convolution__", "__roipooling__", "__pooling__", "__unpooling__", "__batch_normalization__", "__optimized_rnnstack__", "__clip__", "__element_select__", "__splice__", "__combine__", "__alias__", "__as_block__", "__is_first__", "__is_last__", "__first__", "__last__", "__where__", "__gather__", "__scatter__", "__broadcast_as__"
