# setup i3.metal node
``` bash
sudo apt update
sudo apt install build-essential
sudo mdadm --create /dev/md0 --auto md --level=0 --raid-devices=8 /dev/nvme{1..8}n1
sudo chmod 777 /dev/md0
sudo apt install fio
```


<!--
 gpu node old
sudo apt-get install linux-headers-$(uname -r)
sudo apt install cuda-toolkit-12 libnuma-dev &&
sudo apt-key del 7fa2af80 &&
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb &&
sudo dpkg -i cuda-keyring_1.1-1_all.deb &&
sudo apt update &&
sudo apt install cuda-toolkit-12 nvidia-gds &&
-->

# setup g6.8xlarge node
``` bash
sudo apt update
sudo apt install build-essential
sudo apt-get install linux-headers-$(uname -r)
sudo apt-key del 7fa2af80 &&
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb &&
sudo dpkg -i cuda-keyring_1.1-1_all.deb &&
sudo apt update &&
sudo apt install libnuma-dev cuda-toolkit-12 # this needs to be done separetely to next line!
sudo apt install nvidia-gds &&
sudo apt install nvidia-driver-535 nvidia-utils-535 &&

sudo mdadm --create /dev/md0 --auto md --level=0 --raid-devices=2 /dev/nvme{1..2}n1
sudo chmod 777 /dev/md0

```

export PATH="~/golap/third_party/cmake-3.26.4/bin:$PATH"

_CUDA_PATH="/usr/local/cuda-12"
export CUDACXX="$_CUDA_PATH/bin/nvcc"
#export nvcc_path="$_CUDA_PATH/bin/nvcc"
export CPATH=$_CUDA_PATH/include:$CPATH
export LD_LIBRARY_PATH=$_CUDA_PATH/targets/x86_64-linux/lib/:$LD_LIBRARY_PATH
export PATH="$_CUDA_PATH/bin:$_CUDA_PATH/gds/tools/:$PATH"
export CXX=/usr/bin/g++-11

``` bash
cd golap
# setup cmake first, then init.sh
./scripts/init.sh
# fix a problem with nvcomp, -L/usr/local/cuda/lib64
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release .. && make -j ssb ssb_disk_db
```

# fio
``` bash
fio --filesize=16GB --filename=/dev/md0 --io_size=16GB --name=bla --rw=read --iodepth=32 --ioengine=libaio --direct=1 --blocksize=16k --numjobs=16
```
