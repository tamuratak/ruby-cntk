require "test/unit"
require "cntk"
require "numo/narray"
require "pp"

class TestCNTKReader < Test::Unit::TestCase
  include CNTK
  include CNTK::Ops

  def test_text_reader
    dict = { 
      epochSize: 10,
      verbosity: 0,
      randomize: false,
      randomizationWindow: 30,
      deserializers: 
      [{ 
         type: "CNTKTextFormatDeserializer",
         file: File.join(File.dirname(__FILE__), "x3y1.txt"),
         maxErrors: 100,
         skipSequenceIds: false,
         traceLevel: 0,
         input: {
           x: { dim: 3, format: "dense"},
           y: { dim: 1, format: "dense"},
         }
       }]
    }
    batch = CNTK.create_composite_minibatch_source(dict)
    batch_data = batch.get_next_minibatch(2)
    batch_data.each{|k, v| 
      if k.name == "x"
#        p v.data.to_narray
#        p v.number_of_samples
      end
    }
  end

  def test_dict
    v = StdVectorDictionaryValue.new
    v[0] = DictionaryValue.create(1)
    DictionaryValue.create(v)
  end
end
