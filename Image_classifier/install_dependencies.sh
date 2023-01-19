if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    sudo apt install cmake
    sudo apt install libboost-all-dev
    sudo apt install ocl-icd-opencl-dev
    sudo apt-get install xorg-dev
    sudo apt-get install libglfw3
    sudo apt-get install libglfw3-dev

    # Potentially not needed but I am not sure.
    sudo apt-get install libglew-dev
    sudo apt-get install libglu1-mesa-dev
elif [[ "$OSTYPE" == "darwin"* ]]; then
    brew install cmake
    brew install glfw
    brew install boost
    brew install ocl-icd
else
    echo -n "Unsupported operating system. Use Linux or MacOS"
fi