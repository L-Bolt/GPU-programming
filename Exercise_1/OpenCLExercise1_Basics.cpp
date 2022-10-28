//////////////////////////////////////////////////////////////////////////////
// OpenCL exercise 1: Basics
//////////////////////////////////////////////////////////////////////////////

// includes
#include <stdio.h>

#include <Core/Assert.hpp>
#include <Core/Time.hpp>
#include <OpenCL/cl-patched.hpp>
#include <OpenCL/Program.hpp>
#include <OpenCL/Event.hpp>
#include <OpenCL/Device.hpp>

#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>

//////////////////////////////////////////////////////////////////////////////
// CPU implementation
//////////////////////////////////////////////////////////////////////////////
void calculateHost (const std::vector<float>& h_input, std::vector<float>& h_output) {
	for (std::size_t i = 0; i < h_output.size (); i++)
		h_output[i] = std::cos (h_input[i]);
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

	// Get the first device of the context
	std::cout << "Context has " << context.getInfo<CL_CONTEXT_DEVICES>().size() << " devices" << std::endl;
	cl::Device device = context.getInfo<CL_CONTEXT_DEVICES>()[0];
	std::vector<cl::Device> devices;
	devices.push_back(device);
	OpenCL::printDeviceInfo(std::cout, device);

	// Create a command queue
	cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);

	// Load the source code
	cl::Program program = OpenCL::loadProgramSource(context, "/home/lb/Desktop/Exercise 1/For Linux/Opencl-Basics-Linux/src/OpenCLExercise1_Basics.cl");
	// Compile the source code. This is similar to program.build(devices) but will print more detailed error messages
	OpenCL::buildProgram(program, devices);
	std::cout << "program built" << std::endl;

	// Create a kernel object
	cl::Kernel kernel1(program, "kernel1");

	// Declare some values
	std::size_t wgSize = 128; // Number of work items per work group
	std::size_t count = wgSize * 100000; // Overall number of work items = Number of elements
	std::size_t size = count * sizeof (float); // Size of data in bytes

	// Allocate space for input data and for output data from CPU and GPU on the host
	std::vector<float> h_input (count);
	std::vector<float> h_outputCpu (count);
	std::vector<float> h_outputGpu (count);

	// Allocate space for input and output data on the device
	//TODO
	cl::Buffer d_input(context, CL_MEM_READ_ONLY, size, NULL, NULL);
	cl::Buffer d_output(context, CL_MEM_WRITE_ONLY, size, NULL, NULL);

	// Initialize memory to 0xff (useful for debugging because otherwise GPU memory will contain information from last execution)
	memset(h_input.data(), 255, size);
	memset(h_outputCpu.data(), 255, size);
	memset(h_outputGpu.data(), 255, size);

	// Initialize input data with more or less random values
	for (std::size_t i = 0; i < count; i++)
		h_input[i] = ((i * 1009) % 31) * 0.1;

	// Do calculation on the host side
	Core::TimeSpan cpu_start = Core::getCurrentTime();
	calculateHost(h_input, h_outputCpu);
	Core::TimeSpan cpu_end = Core::getCurrentTime();

	// Copy input data to device
	//TODO: enqueueWriteBuffer()
	Core::TimeSpan memory_start = Core::getCurrentTime();
	queue.enqueueWriteBuffer(d_input, CL_TRUE, 0, size, h_input.data(), NULL, NULL);

	// Launch kernel on the device
	//TODO
	kernel1.setArg(0, d_input);
	kernel1.setArg(1, d_output);
	kernel1.setArg(2, count);
	Core::TimeSpan memory_end = Core::getCurrentTime();

	cl::Event kernel_event; // Event to measure Kernel time
	queue.enqueueNDRangeKernel(kernel1, 0, count, wgSize, NULL, &kernel_event);

	// Copy output data back to host
	//TODO: enqueueReadBuffer()
	queue.enqueueReadBuffer(d_output, CL_TRUE, 0, size, h_outputGpu.data(), NULL, NULL);

	// Print performance data
	//TODO
	Core::TimeSpan GPU_time = OpenCL::getElapsedTime(kernel_event);
	Core::TimeSpan sequential_time = cpu_end - cpu_start;
	Core::TimeSpan paralellized_time = GPU_time + (memory_end - memory_start);
	std::cout << "CPU time: " << sequential_time << std::endl;
	std::cout << "GPU time: " << paralellized_time << std::endl;
	std::cout << "Speedup: " << sequential_time.getSeconds() / paralellized_time.getSeconds() << std::endl;

	// Check whether results are correct
	std::size_t errorCount = 0;
	for (std::size_t i = 0; i < count; i++) {
		// Allow small differences between CPU and GPU results (due to different rounding behavior)
		if (!(std::abs (h_outputCpu[i] - h_outputGpu[i]) <= 10e-5)) {
			if (errorCount < 15)
				std::cout << "Result for " << i << " is incorrect: GPU value is " << h_outputGpu[i] << ", CPU value is " << h_outputCpu[i] << std::endl;
			else if (errorCount == 15)
				std::cout << "..." << std::endl;
			errorCount++;
		}
	}
	if (errorCount != 0) {
		std::cout << "Found " << errorCount << " incorrect results" << std::endl;
		return 1;
	}

	std::cout << "Success" << std::endl;

	return 0;
}
