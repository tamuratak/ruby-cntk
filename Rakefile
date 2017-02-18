require "rake/extensiontask"
require 'rake/testtask'
require 'rake/clean'

vlibs = ["CNTK"]
other_libs = []
elibs = vlibs + other_libs

pat = "{*/*,*}.{cxx,i,hpp,h,inc}"
swig_pat = "{*/*,*}.{i,hpp,h}"

elibs.each{|s|
  Rake::ExtensionTask.new s do |ext|
    ext.ext_dir = "ext/#{s.downcase}"
    ext.lib_dir = "lib/cntk"
    ext.source_pattern = pat
  end
}

task :cntk => ["ext/cntk/cntk_wrap.cxx"]
file "ext/cntk/cntk_wrap.cxx" => Dir.glob("ext/cntk/#{swig_pat}") do
  Rake::Task["swg:cntk"].execute
end

task :ocl => ["ext/ocl/ocl_wrap.inc"]
file "ext/ocl/ocl_wrap.inc" => Dir.glob("ext/ocl/#{swig_pat}") do
   Rake::Task["swg:ocl"].execute
end

namespace :swg do
  task "cntk" do
    sh "swig -c++ -ruby -Wextra -module CNTK ext/cntk/cntk.i"
  end
end

Rake::TestTask.new do |t|
  t.libs << 'test'
end
desc "Run tests"

task :swig => [:cntk]
task :default => [:test]
task :build => ["swg:cntk", :compile]
