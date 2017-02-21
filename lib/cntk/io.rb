module CNTK
  def create_composite_minibatch_source(dict)    
    if dict.respond_to?(:to_hash)
      h = {}
      dict.to_hash.each_pair{|k, v|
        k = k.to_s if k.is_a?(Symbol)
        h[k] = v
      }
      des = h["deserializers"]
      unless des.respond_to?(:to_ary)
        h["deserializers"] = [des]
      end
      dict = Dictionary.create(h)
    end
    __create_composite_minibatch_source__(dict)
  end
end
