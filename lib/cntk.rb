require "cntk/CNTK"
require "cntk/axis"
require "cntk/dictionary"
require "cntk/function"
require "cntk/inspect"
require "cntk/io"
require "cntk/layers"
require "cntk/learner"
require "cntk/ndmask"
require "cntk/ndshape"
require "cntk/ndarrayview"
require "cntk/ops"
require "cntk/trainer"
require "cntk/value"
require "cntk/variable"

module CNTK
  remove_method( *private_instance_methods(false) )
end
