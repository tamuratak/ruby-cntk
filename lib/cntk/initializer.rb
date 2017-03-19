module CNTK
module Initializer
class << self
  
  def constant
    CNTK.__constant_initializer__
  end

  def uniform(scale, seed = CNTK.SentinelValueForAutoSelectRandomSeed)
      CNTK.__uniform_initializer__(scale, seed)
  end

  def normal(scale,
             output_rank: CNTK.SentinelValueForInferParamInitRank,
             filter_rank: CNTK.SentinelValueForInferParamInitRank,
             seed:        CNTK.SentinelValueForAutoSelectRandomSeed)
    CNTK.__normal_initializer__(scale, output_rank, filter_rank, seed)
  end

  def glorot_uniform(scale = CNTK.DefaultParamInitScale,
                     output_rank: CNTK.SentinelValueForInferParamInitRank,
                     filter_rank: CNTK.SentinelValueForInferParamInitRank,
                     seed:        CNTK.SentinelValueForAutoSelectRandomSeed)
    CNTK.__glorot_uniform_initializer__(scale, output_rank, filter_rank, seed)
  end

  def glorot_normal(scale = CNTK.DefaultParamInitScale,
                    output_rank: CNTK.SentinelValueForInferParamInitRank,
                    filter_rank: CNTK.SentinelValueForInferParamInitRank,
                    seed:        CNTK.SentinelValueForAutoSelectRandomSeed)
    CNTK.__glorot_normal_initializer__(scale, output_rank, filter_rank, seed)
  end

  def xavier(scale,
             output_rank: CNTK.SentinelValueForInferParamInitRank,
             filter_rank: CNTK.SentinelValueForInferParamInitRank,
             seed:        CNTK.SentinelValueForAutoSelectRandomSeed)
    CNTK.__xavier_initializer__(scale, output_rank, filter_rank, seed)
  end

  def he_uniform(scale = CNTK.DefaultParamInitScale,
                 output_rank: CNTK.SentinelValueForInferParamInitRank,
                 filter_rank: CNTK.SentinelValueForInferParamInitRank,
                 seed:        CNTK.SentinelValueForAutoSelectRandomSeed)
    CNTK.__he_uniform_initializer__(scale, output_rank, filter_rank, seed)
  end

  def he_normal(scale = CNTK.DefaultParamInitScale,
                output_rank: CNTK.SentinelValueForInferParamInitRank,
                filter_rank: CNTK.SentinelValueForInferParamInitRank,
                seed:        CNTK.SentinelValueForAutoSelectRandomSeed)
    CNTK.__he_normal_initializer__(scale, output_rank, filter_rank, seed)
  end

  def bilinear(kernel_width, kernel_height)
    CNTK.__bilinear_initializer__(kernel_width, kernel_height)
  end

  def initializer_with_rank(initializer,
                            output_rank: CNTK.SentinelValueForInferParamInitRank,
                            filter_rank: CNTK.SentinelValueForInferParamInitRank)
    CNTK.__random_initializer_with_rank__(initializer, output_rank, filter_rank)
  end

end # class << self
end # module Initializer
end # module CNT
