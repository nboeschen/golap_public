#!/bin/bash
# go to third_party dir
THE_DIR="$(dirname $(readlink -f "$0"))/../third_party"
mkdir -p $THE_DIR &&
cd $THE_DIR &&

# repo_init="submodule"
repo_init="explicit"

if [[ "$repo_init" == "submodule" ]]; then
    # update submodules
    git submodule update --init --recursive
else
    git clone "https://github.com/lz4/lz4" lz4 2> /dev/null || (cd lz4; git checkout dev; git pull; cd ..)
    git clone "https://github.com/gflags/gflags" gflags 2> /dev/null || (cd gflags; git checkout master; git pull; cd ..)
    git clone "https://github.com/catchorg/Catch2" Catch2 2> /dev/null || (cd Catch2; git checkout devel; git pull; cd ..)
    git clone "https://github.com/NVIDIA/nvcomp" nvcomp 2> /dev/null || (cd nvcomp; git checkout main; git pull; cd ..)
    git clone "https://github.com/cameron314/concurrentqueue" concurrentqueue 2> /dev/null || (cd concurrentqueue; git checkout main; git pull; cd ..)
    git clone "https://github.com/google/snappy" snappy 2> /dev/null || (cd snappy; git checkout main; git pull; cd ..)
    git clone "https://github.com/viktorleis/perfevent/" perfevent 2> /dev/null || (cd perfevent; git checkout main; git pull; cd ..)
fi

# install cmake 3.26.4
cd $THE_DIR &&
wget --continue https://github.com/Kitware/CMake/releases/download/v3.26.4/cmake-3.26.4-linux-x86_64.tar.gz &&
tar -xzf cmake-3.26.4-linux-x86_64.tar.gz && rm cmake-3.26.4-linux-x86_64.tar.gz &&
mv cmake-3.26.4-linux-x86_64 cmake-3.26.4
# echo "export PATH=$THE_DIR/cmake-3.26.4/bin:\$PATH" >> ~/.bashrc 
# && source ~/.bashrc

# install lz4
cd "$THE_DIR/lz4" && git checkout ce8ee024b240befac4ca8ab12c6cd812f4a7e38b &&
mkdir -p install &&
cd build && 
cmake cmake/ && make -j && make DESTDIR=../install install &&

# install gflags
cd "$THE_DIR/gflags" && git checkout 986e8eed00ded8168ef4eaa6f925dc6be50b40fa &&
mkdir -p build && mkdir -p install &&
cd build &&
cmake -DCMAKE_INSTALL_PREFIX=../install/ .. &&
make -j && make install

# cd "$THE_DIR/Catch2" && git checkout 4acc51828f7f93f3b2058a63f54d112af4034503

# concurrentqueue
cd "$THE_DIR/concurrentqueue" && git checkout 65d6970912fc3f6bb62d80edf95ca30e0df85137 && sed -i "s/BLOCK_SIZE/CQBLOCK_SIZE/g" concurrentqueue.h # fix colliding constexpr

# snappy
cd "$THE_DIR/snappy" && git checkout 984b191f0fefdeb17050b42a90b7625999c13b8d &&
git submodule update --init && mkdir -p build && cd build && cmake .. && make -j


# install duckdb 0.7.1
# cd $THE_DIR &&
# wget https://github.com/duckdb/duckdb/releases/download/v0.7.1/libduckdb-linux-amd64.zip && mkdir -p duckdb &&
# unzip libduckdb-linux-amd64.zip -d duckdb &&
# rm libduckdb-linux-amd64.zip

# alternative build duckdb from source:
# cd $THE_DIR &&
# wget https://github.com/duckdb/duckdb/releases/download/v0.7.1/libduckdb-src.zip &&
# unzip libduckdb-src.zip -d duckdb_src && rm libduckdb-src.zip

# download nvcomp extensions and install nvcomp with them
cd "$THE_DIR/nvcomp" && git checkout a6e4e64a177e07cd2e5c8c5e07bb66ffefceae84 &&
wget --continue http://developer.download.nvidia.com/compute/nvcomp/2.2/local_installers/nvcomp_exts_x86_64_ubuntu20.04-2.2.tar.gz &&
tar -xzf nvcomp_exts_x86_64_ubuntu20.04-2.2.tar.gz &&
mkdir -p build && mkdir -p install &&
cd build &&
cmake -DNVCOMP_EXTS_ROOT=../ubuntu20.04/11.0 -DCMAKE_INSTALL_PREFIX=../install .. &&
make -j && make install
