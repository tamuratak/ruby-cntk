require 'mkmf'

have_library("c++") or have_library("stdc++")

dir_config("cntklibrary-2.0")
have_library("cntklibrary-2.0")

# rake2.3 compile -- --with-cntklibrary-2.0-lib=/cntk/build/cpu/release/lib/ --with-cntklibrary-2.0-include=/cntk/Source/CNTKv2LibraryDll/API/
# rake2.3 compile -- --with-cntklibrary-2.0-lib=/cntk/cntk/lib/ --with-cntklibrary-2.0-include=/cntk/Include/

$CXXFLAGS = ($CXXFLAGS || "") + " -std=c++11 -O2 -DSWIG -Wl,--whole-archive " 

create_makefile('cntk/CNTK')
