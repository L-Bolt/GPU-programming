if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    sudo apt install cmake
    sudo apt install libboost-all-dev
    sudo apt install ocl-icd-opencl-dev
    sudo apt-get install xorg-dev
    sudo apt-get install libglfw3
    sudo apt-get install libglfw3-dev
elif [[ "$OSTYPE" == "darwin"* ]]; then
    brew install cmake
    brew install glfw
    brew install boost
    brew install ocl-icd
