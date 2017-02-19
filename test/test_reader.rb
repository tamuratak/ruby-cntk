require "test/unit"
require "cntk"
require "numo/narray"

class TestCNTKReader < Test::Unit::TestCase
  include CNTK
  include CNTK::Ops

  def test_text_reader
    dict = Dictionary.create({epochSize: 2,
                               verbosity: 0,
                               randomize: true,
                               randomizationWindow: 30,
                               deserializers: [{ 
                                                 type: "CNTKTextFormatDeserializer",
                                                 "module" => "CNTKTextFormatReader",
                                                 file: "/home/ubuntu/src/github/ruby-cntk/test/x3y1.txt",
                                                 maxErrors: 100,
                                                 skipSequenceIds: false,
                                                 traceLevel: 2,
                                                 input: {
                                                   x: { dim: 3, format: "dense"},
                                                   y: { dim: 1, format: "dense"},
                                                 }
                                               }]
                              }) 
#    Dictionary.load("/home/ubuntu/src/github/ruby-cntk/test/reader.config")
    p dict["x"].value_type

    __create_composite_minibatch_source__(dict)
  end

  def test_dict
    v = StdVectorDictionaryValue.new
    v[0] = DictionaryValue.create(1)
    DictionaryValue.create(v)
  end
end
