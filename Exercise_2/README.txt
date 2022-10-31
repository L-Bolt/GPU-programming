-----------------------------------------
For Linux Users:
-----------------------------------------

1. Unzip the archive in server or in your local folder.
unzip -r file.zip .      // will extract the file.zip to current directory,
			    change the "." to final location, in case you want to save it somewhere else
2. Create a "build" folder inside the Exercise folder you have unzipped.
   mkdir build
3. Go to build folder:
   cd build
4. Run the cmake inside the build folder:
   cmake ..
5. Compile the code with make command:
   make
6. run the executable file which will be named by the project name which you have
   written in CMakeLists.txt
   For example: ./project_name


------------------------------------------
For Windows
------------------------------------------

1. Copy files from "src" folder in the zip to the src folder of the first exercise.
2. You do not need the rest of files from Exercise2-GPU.zip file
3. Use the Exercise 1 project in Visual Studio as a base.
4. Change the cpp file name in CMakeLists.txt in add_executable() to OpenCLExercise2_Mandelbrot.cpp
5. In the OpenCLExercise2_Mandelbrot.cpp change the path to the cl file to the new one.
6. Done

Note: in case you have problems with IoStream Error it relates to wrong path for .cl file.

