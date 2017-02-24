module CNTK
  LearningRateSchedule = MomentumSchedule
  MinibatchSizeSchedule = TrainingParameterPerSampleSchedule

  # TrainingParameterPerSampleSchedule    == MinibatchSizeSchedule 
  # TrainingParameterPerMinibatchSchedule 
  #

  module_function

  def training_parameter_schedule(schedule, unit, epoch_size = nil)

    case unit
    when :sample
      klass = TrainingParameterPerSampleSchedule
    when :minibatch
      klass = TrainingParameterPerMinibatchSchedule
    else
      raise "unknown unit"
    end

    if schedule.is_a?(Numeric) 
      unless epoch_size.nil?
        raise "epoch_size can't be given when schedule is Numeric."
      else
        klass.new(schedule)
      end
    else
      if epoch_size.nil?
        klass.new(schedule)
      else
        klass.new(schedule, epoch_size)
      end
    end

  end

  def momentum_schedule(schedule, unit = :minibatch, epoch_size = nil)
    training_parameter_schedule(schedule, unit, epoch_size)
  end

  def momentum_as_time_constant_schedule(schedule, epoch_size)
    klass = MomentumAsTimeConstantSchedule
    if schedule.is_a?(Numeric) 
      if epoch_size.nil?
        raise "epoch_size can't be given when schedule is Numeric."
      else
        klass.new(schedule)
      end
    else
      if epoch_size.nil?
        klass.new(schedule)
      else
        klass.new(schedule, epoch_size)
      end
    end
  end
  
  def sgd_learner(parameters, lr, l1_weight: 0.0, l2_weight: 0.0, 
                  std_dev: 0.0, threshold: Float::INFINITY, truncation: true)

    ga = training_parameter_schedule(std_dev, :minibatch)

    opt = AdditionalLearningOptions.new
    opt.l1_regularization_weight = l1_weight
    opt.l2_regularization_weight = l2_weight
    opt.gaussian_noise_injection_std_dev = ga
    opt.gradient_clipping_threshold_per_sample = threshold
    opt.gradient_clipping_with_truncation = truncation

    __sgdlearner__(parameters, lr, opt)
  end

  def momentum_sgd_learner(parameters, lr, momentum, unit_gain: CNTK.default_unit_gain_value(),
                           l1_weight: 0.0, l2_weight: 0.0, 
                           std_dev: 0.0, threshold: Float::INFINITY, truncation: true)

    opt = AdditionalLearningOptions.new
    opt.l1_regularization_weight = l1_weight
    opt.l2_regularization_weight = l2_weight
    opt.gaussian_noise_injection_std_dev = ga
    opt.gradient_clipping_threshold_per_sample = threshold
    opt.gradient_clipping_with_truncation = truncation

    __momentum_sgd_learner__(parameters, lr, momentum, unit_gain, opt)
  end

  
end
