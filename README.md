### 说明
本环境配置步骤来自[ml-neuman的dockerfile](https://github.com/apple/ml-neuman/tree/main/preprocess "ml-neuman的dockerfile")，建议两个一同参考。<br />
总而言之，你需要colmap，ROMP，4D-humans<br />
可以在ubuntu22上跑，其他应该也可以<br />
建议一块一块执行，或者一行一行执行。<br />

### 安装一坨依赖
```bash
apt-get update && apt-get install -y \
    git \
    cmake \
    vim \
    wget \
    unzip \
    build-essential \
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libboost-test-dev \
    libeigen3-dev \
    libsuitesparse-dev \
    libfreeimage-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    libcgal-qt5-dev
```

### Build and install ceres solver

```bash
apt-get -y install \
libatlas-base-dev \
libsuitesparse-dev
git clone https://github.com/ceres-solver/ceres-solver.git --branch 1.14.0 #基本上所有的git都需要中国大陆用户自行添加代理
cd ceres-solver
mkdir build
cd build
cmake .. -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF \  #ubuntu22在这里可能出现的问题以及解决方法：https://github.com/ceres-solver/ceres-solver/issues/669
make -j12 #根据设备水平修改-j4 -> j24
make install
```

### Build and install COLMAP
原作者注释：
Note: This Dockerfile has been tested using COLMAP pre-release 3.6. Later versions of COLMAP (which will be automatically cloned as default) may have problems using the environment described thus far. If you encounter
problems and want to install the tested release, then uncomment the branch specification in the line below
```bash
git clone https://github.com/colmap/colmap.git #--branch 3.6
cd colmap
git checkout 96d4ba0b55c0d1f98c8c432420ecd6540868c398 
mkdir build
cd build
#ubuntu22安装colmap可能出现的问题以及解决方法：https://github.com/colmap/colmap/issues/1626
cmake .. -DCUDA_NVCC_FLAGS="--std c++14"
make -j24 
make install
```

### 下载ROMP和detectron2
ROMP和detecron的路径可以变，都安装在conda虚拟环境ROMP里即可。如果你修改了路径，gen_run.py里的路径也需要修改
```bash
cd /ROMP
git clone --recurse-submodules https://github.com/jiangwei221/ROMP.git
git checkout f1aaf0c1d90435bbeabe39cf04b15e12906c6111

git clone --recurse-submodules https://github.com/jiangwei221/detectron2.git
cd detectron2
git checkout 2048058b6790869e5add8832db2c90c556c24a3e
```
### 安装ROMP和detectron2
请注意，可以根据你的ubuntu和显卡版本安装pytorch <br />
以下是配置ROMP 和detectron2的environment，参考了neuman的issue（你在安装这一部分时有问题也可以参考issue） <br />
```bash
cd ROMP
conda create -n ROMP python==3.8.8 && \
conda activate ROMP && \
conda install -c pytorch pytorch=1.11 torchvision torchaudio cudatoolkit=11.3 && \ #实际上我是cuda11.8，这里用11.3也安装上了
conda install -c fvcore -c iopath -c conda-forge fvcore iopath && \
conda install -c bottler nvidiacub && \
onda install pytorch3d -c pytorch3d

conda activate ROMP && \
cd /preprocessForHugs/ROMP && \
pip install -r requirements.txt && \
pip install av

python -m pip install -e detectron2 && \
pip install setuptools==59.5.0
```

### 配置preprocessForHugs
```bash
conda create -n preprocessForHugs python=3.10 -y && \
conda activate preprocessForHugs && \
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118 # 根据你的设备修改
conda install -c conda-forge igl && \
pip install opencv-python joblib open3d imageio tensorboardX chumpy lpips scikit-image ipython matplotlib
```


### 安装4d-human
[4D-human](https://github.com/shubham-goel/4D-Humans?tab=readme-ov-file "4D-human")的安装
```bash
git clone https://github.com/shubham-goel/4D-Humans.git
cd 4D-Humans
conda create --name 4D-humans python=3.10
conda activate 4D-humans
pip install torch #不一定是cu12，根据自己硬件装
pip install -e .[all]
pip install git+https://github.com/brjathu/PHALP.git #安装phalp
#可能会有pyrender问题：pyrender报错ImportError: (‘Unable to load EGL library‘, ‘EGL: cannot open shared object file:
如何解决：https://blog.csdn.net/jiaoooooo/article/details/133500112
```