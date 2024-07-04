# With python:3.8 docker image

sudo apt-get update -y
sudo apt-get install build-essential -y
sudo apt install curl git -y

if ! command -v micromamba &> /dev/null
then
  echo "micromamba could not be found"
  curl -L micro.mamba.pm/install.sh | bash
  source ~/.bashrc
fi

if ! command -v cmake &> /dev/null
then
  echo "cmake could not be found"
  sudo apt install cmake -y
  sudo apt-get install zlib1g-dev -y  # https://stackoverflow.com/questions/24808150/how-to-point-cmake-to-zlib-include-path
fi

if ! { micromamba env list | grep 'pt20'; } >/dev/null 2>&1; then
    echo "Creating environment"
    micromamba create -n pt20 python=3.8 -c conda-forge -y
    micromamba activate pt20
    source remote/setup.sh

    pip install ipykernel
    python -m ipykernel install --user --name=pt20
else
    echo "Found pre-existing environment. Activating it."
    micromamba activate pt20
fi

# was needed for pytorch 2.0 at some point, revisit
export C_INCLUDE_PATH=$HOME/micromamba/envs/pt20/lib/python3.8/site-packages/triton/third_party/cuda/include/

# # --------------------------------

# Install cnpy
mkdir -p code/data/cython/build
cd code/data/cython/build
cmake ../cnpy
make
sudo make install
cd ../../../..
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/code/data/cython/build
export LD_LIBRARY_PATH

# Build cython modules in data
cd code/data/cython
python setup.py build_ext --inplace
cd ../../..

# Build cython modules in xclib
cd code/main/xclib
python setup.py build_ext --inplace
cd ../../..
