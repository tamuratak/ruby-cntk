module CNTK
class Trainer

  def self.create_trainer(*args)
    CNTK.__create_trainer__(*args)
  end

end
end
