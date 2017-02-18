Gem::Specification.new do |s|
  s.name        = 'ruby-cntk'
  s.version     = '0.1.0.pre1'
  s.date        = '2017-02-16'
  s.summary     = "Ruby bindings for Microsoft CNTK, an open source deep-learning toolkit"
  s.description = "Ruby bindings for Microsoft Cognitive Toolkit (CNTK), an open source deep-learning toolkit"
  s.authors     = ["Takashi Tamura"]
  s.email       = ''
  s.files       = ["LICENSE", 
#                   "README.md",
                   "lib/cntk.rb",                   
                   "ext/cntk/cntk_wrap.cxx",
                   "ext/cntk/extconf.rb"] + Dir.glob("lib/cntk/**/*.rb")
  s.extensions  = ["ext/cntk/extconf.rb"]
  s.homepage    = 'https://github.com/tamuratak/ruby-cntk'
  s.license     = 'MIT'
  s.add_runtime_dependency 'rake-compiler', '~> 0.9.5'
  s.rdoc_options << "--exclude=."
  s.requirements << 'CNTK >= 2.0b11'
  s.required_ruby_version = '>= 2.3.0'
end
