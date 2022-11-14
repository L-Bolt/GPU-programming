//////////////////////////////////////////////////////////////////////////////
// OpenCL exercise 3: Sobel filter
//////////////////////////////////////////////////////////////////////////////

// includes
#include <stdio.h>

#include <Core/Assert.hpp>
#include <Core/Time.hpp>
#include <Core/Image.hpp>
#include <OpenCL/cl-patched.hpp>
#include <OpenCL/Program.hpp>
#include <OpenCL/Event.hpp>
#include <OpenCL/Device.hpp>

#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>

#include <boost/lexical_cast.hpp>

//////////////////////////////////////////////////////////////////////////////
// CPU implementation
//////////////////////////////////////////////////////////////////////////////
int getIndexGlobal(std::size_t countX, int i, int j) {
	return j * countX + i;
}
// Read value from global array a, return 0 if outside image
float getValueGlobal(const std::vector<float>& a, std::size_t countX, std::size_t countY, int i, int j) {
	if (i < 0 || (size_t) i >= countX || j < 0 || (size_t) j >= countY)
		return 0;
	else
		return a[getIndexGlobal(countX, i, j)];
}
void sobelHost(const std::vector<float>& h_input, std::vector<float>& h_outputCpu, std::size_t countX, std::size_t countY) {
	for (int i = 0; i < (int) countX; i++) {
		for (int j = 0; j < (int) countY; j++) {
			float Gx = getValueGlobal(h_input, countX, countY, i-1, j-1)+2*getValueGlobal(h_input, countX, countY, i-1, j)+getValueGlobal(h_input, countX, countY, i-1, j+1)
					-getValueGlobal(h_input, countX, countY, i+1, j-1)-2*getValueGlobal(h_input, countX, countY, i+1, j)-getValueGlobal(h_input, countX, countY, i+1, j+1);
			float Gy = getValueGlobal(h_input, countX, countY, i-1, j-1)+2*getValueGlobal(h_input, countX, countY, i, j-1)+getValueGlobal(h_input, countX, countY, i+1, j-1)
					-getValueGlobal(h_input, countX, countY, i-1, j+1)-2*getValueGlobal(h_input, countX, countY, i, j+1)-getValueGlobal(h_input, countX, countY, i+1, j+1);
			h_outputCpu[getIndexGlobal(countX, i, j)] = sqrt(Gx * Gx + Gy * Gy);
		}
	}
}

//////////////////////////////////////////////////////////////////////////////
// Main function
//////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) {
	// Create a context
	//cl::Context context(CL_DEVICE_TYPE_GPU);
	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);
	if (platforms.size() == 0) {
		std::cerr << "No platforms found" << std::endl;
		return 1;
	}
	int platformId = 0;
	for (size_t i = 0; i < platforms.size(); i++) {
		if (platforms[i].getInfo<CL_PLATFORM_NAME>() == "AMD Accelerated Parallel Processing") {
			platformId = i;
			break;
		}
	}
	cl_context_properties prop[4] = { CL_CONTEXT_PLATFORM, (cl_context_properties) platforms[platformId] (), 0, 0 };
	std::cout << "Using platform '" << platforms[platformId].getInfo<CL_PLATFORM_NAME>() << "' from '" << platforms[platformId].getInfo<CL_PLATFORM_VENDOR>() << "'" << std::endl;
	cl::Context context(CL_DEVICE_TYPE_GPU, prop);

	// Get a device of the context
	int deviceNr = argc < 2 ? 1 : atoi(argv[1]);
	std::cout << "Using device " << deviceNr << " / " << context.getInfo<CL_CONTEXT_DEVICES>().size() << std::endl;
	ASSERT (deviceNr > 0);
	ASSERT ((size_t) deviceNr <= context.getInfo<CL_CONTEXT_DEVICES>().size());
	cl::Device device = context.getInfo<CL_CONTEXT_DEVICES>()[deviceNr - 1];
	std::vector<cl::Device> devices;
	devices.push_back(device);
	OpenCL::printDeviceInfo(std::cout, device);

	// Create a command queue
	cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);

	// Load the source code
	cl::Program program = OpenCL::loadProgramSource(context, "/home/lb/Desktop/GPU-programming/Exercise_3/src/OpenCLExercise3_Sobel.cl");
	// Compile the source code. This is similar to program.build(devices) but will print more detailed error messages
	OpenCL::buildProgram(program, devices);

	// Declare some values
	std::size_t wgSizeX = 16; // Number of work items per work group in X direction
	std::size_t wgSizeY = 16;
	std::size_t countX = wgSizeX * 40; // Overall number of work items in X direction = Number of elements in X direction
	std::size_t countY = wgSizeY * 30;
	//countX *= 3; countY *= 3;
	std::size_t count = countX * countY; // Overall number of elements
	std::size_t size = count * sizeof (float); // Size of data in bytes

	// Allocate space for output data from CPU and GPU on the host
	std::vector<float> h_input (count);
	std::vector<float> h_outputCpu (count);
	std::vector<float> h_outputGpu (count);

	// Allocate space for input and output data on the device
	cl::Buffer d_input(context, CL_MEM_READ_ONLY, size, NULL, NULL);
	cl::Buffer d_output(context, CL_MEM_WRITE_ONLY, size, NULL, NULL);

	// Initialize memory to 0xff (useful for debugging because otherwise GPU memory will contain information from last execution)
	memset(h_input.data(), 255, size);
	memset(h_outputCpu.data(), 255, size);
	memset(h_outputGpu.data(), 255, size);
	//TODO: GPU

	//////// Load input data ////////////////////////////////
	// Use random input data
	/*
	for (int i = 0; i < count; i++)
		h_input[i] = (rand() % 100) / 5.0f - 10.0f;
	*/
	// Use an image (Valve.pgm) as input data

	std::vector<float> inputData;
	std::size_t inputWidth, inputHeight;
	Core::readImagePGM("/home/lb/Desktop/GPU-programming/Exercise_3/src/Valve.pgm", inputData, inputWidth, inputHeight);
	for (size_t j = 0; j < countY; j++) {
		for (size_t i = 0; i < countX; i++) {
			h_input[i + countX * j] = inputData[(i % inputWidth) + inputWidth * (j % inputHeight)];
		}
	}

	// Do calculation on the host side
	Core::TimeSpan cpu_start = Core::getCurrentTime();
	sobelHost(h_input, h_outputCpu, countX, countY);
	Core::TimeSpan cpu_end = Core::getCurrentTime();
	Core::TimeSpan sequential_time = cpu_end - cpu_start;

	//////// Store CPU output image ///////////////////////////////////
	Core::writeImagePGM("output_sobel_cpu.pgm", h_outputCpu, countX, countY);

	std::cout << std::endl;
	// Iterate over all implementations (task 1 - 3)
	for (int impl = 1; impl <= 3; impl++) {
		std::cout << "\033[1;31mImplementation\033[0m " << impl << ":" << std::endl;

		// Reinitialize output memory to 0xff
		memset(h_outputGpu.data(), 255, size);
		//TODO: GPU

		// Define events
		cl::Event host_to_device;
		cl::Event kernel_execution_time;
		cl::Event device_to_host;

		// Create a kernel object
		std::string kernelName = "sobelKernel" + boost::lexical_cast<std::string> (impl);
		cl::Kernel sobelKernel(program, kernelName.c_str());

		// Copy input data to device
		if (impl == 3) {
			// Create buffer for cl::Image2D.
			cl::Image2D d_image(context, CL_MEM_READ_ONLY, cl::ImageFormat(CL_R, CL_FLOAT), inputWidth, inputHeight);

			cl::size_t<3> origin;
			origin[0] = origin[1] = origin[2] = 0;

			cl::size_t<3> region;
			region[0] = inputWidth;
			region[1] = inputHeight;
			region[2] = 1;

			int error = queue.enqueueWriteImage(d_image, CL_TRUE, origin, region, inputWidth * sizeof(float), 0, h_input.data(), NULL, &host_to_device);

			sobelKernel.setArg<cl::Image2D>(0, d_image);
			sobelKernel.setArg<cl::Buffer>(1, d_output);
		}
		else {
			queue.enqueueWriteBuffer(d_input, CL_TRUE, 0, size, h_input.data(), NULL, &host_to_device);

			// Launch kernel on the device
			sobelKernel.setArg(0, d_input);
			sobelKernel.setArg(1, d_output);
		}

		cl::NDRange global_range2D(countX, countY);
		cl::NDRange local_range2D(wgSizeX, wgSizeY);
		int error2 = queue.enqueueNDRangeKernel(sobelKernel, 0, global_range2D, local_range2D, NULL, &kernel_execution_time);

		// Copy output data back to host
		queue.finish();
		queue.enqueueReadBuffer(d_output, CL_TRUE, 0, size, h_outputGpu.data(), NULL, &device_to_host);

		// Print performance data
		Core::TimeSpan GPU_kernel_time = OpenCL::getElapsedTime(kernel_execution_time);
		Core::TimeSpan GPU_memory_time = OpenCL::getElapsedTime(host_to_device) + OpenCL::getElapsedTime(device_to_host);
		Core::TimeSpan paralellized_time = GPU_kernel_time + GPU_memory_time;
		std::cout << "CPU time: " << sequential_time << std::endl;
		std::cout << "GPU time without memory transfer: " << GPU_kernel_time << std::endl;
		std::cout << "GPU time with memory transfer: " << paralellized_time << std::endl;
		std::cout << "Speedup: " << sequential_time.getMilliseconds() / paralellized_time.getMilliseconds() << std::endl;
		std::cout << std::endl;
		std::cout << "MPixel/s on CPU:" << ((inputWidth * inputHeight) / 1000000.0f) / sequential_time.getSeconds() << std::endl;
		std::cout << "MPixel/s on GPU:" << ((inputWidth * inputHeight) / 1000000.0f) / paralellized_time.getSeconds() << std::endl;

		//////// Store GPU output image ///////////////////////////////////
		Core::writeImagePGM("output_sobel_gpu_" + boost::lexical_cast<std::string> (impl) + ".pgm", h_outputGpu, countX, countY);

		// Check whether results are correct
		std::size_t errorCount = 0;
		for (size_t i = 0; i < countX; i = i + 1) { //loop in the x-direction
			for (size_t j = 0; j < countY; j = j + 1) { //loop in the y-direction
				size_t index = i + j * countX;
				// Allow small differences between CPU and GPU results (due to different rounding behavior)
				if (!(std::abs (h_outputCpu[index] - h_outputGpu[index]) <= 1e-5)) {
					if (errorCount < 15)
						std::cout << "Result for " << i << "," << j << " is incorrect: GPU value is " << h_outputGpu[index] << ", CPU value is " << h_outputCpu[index] << std::endl;
					else if (errorCount == 15)
						std::cout << "..." << std::endl;
					errorCount++;
				}
			}
		}
		if (errorCount != 0) {
			std::cout << "Found " << errorCount << " incorrect results" << std::endl;
			return 1;
		}

		std::cout << std::endl;
	}

	std::cout << "Success" << std::endl;

	return 0;
}
