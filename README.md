

    export PATH="$PATH:/usr/local/cuda/bin"
    mkdir -p build
    cd build
    CXX=g++-8 CC=gcc-8 cmake ..
