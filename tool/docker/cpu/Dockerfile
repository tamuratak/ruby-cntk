FROM microsoft/cntk:2.0.beta11.0-cpu-python3.5

RUN apt-get update \
 && apt-get install -y software-properties-common \
 && add-apt-repository -y ppa:brightbox/ruby-ng \
 && apt-get update \
 && apt-get upgrade -y \
 && apt-get install -y \
    build-essential \
    clang-3.8 \
    clang-3.8-doc \
    llvm-3.8-doc \
    cmake \
    git \
    pkg-config \
    ruby2.3 \
    ruby2.3-dev \
    unzip \
    wget \
 && gem2.3 install rake-compiler minitest test-unit numo-narray

RUN gem2.3 install --pre ruby-cntk -- --with-cntklibrary-2.0-lib=/cntk/cntk/lib/ --with-cntklibrary-2.0-include=/cntk/Include/