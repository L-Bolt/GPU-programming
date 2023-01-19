#ifndef GUI
#define GUI

#if defined(__APPLE__)
#define GLFW_INCLUDE_GLCOREARB
#define GL_SILENCE_DEPRECATION
#endif

#include <stdio.h>
#include <string>
#include <iostream>
#include <thread>
#include <future>

#include "Dataset.h"
#include "Cnn.h"
#include "../../lib/imgui/imgui.h"
#include "../../lib/imgui/backends/imgui_impl_glfw.h"
#include "../../lib/imgui/backends/imgui_impl_opengl3.h"

#define GL_SILENCE_DEPRECATION
// #if defined(IMGUI_IMPL_OPENGL_ES2)
// #include <GLES2/gl2.h>
// #endif
#include <GLES2/gl2.h>
#include <GLFW/glfw3.h> // Will drag system OpenGL headers

// [Win32] Our example includes a copy of glfw3.lib pre-compiled with VS2010 to maximize ease of testing and compatibility with old VS compilers.
// To link with VS2010-era libraries, VS2015+ requires linking with legacy_stdio_definitions.lib, which we do using this pragma.
// Your own project should not be affected, as you are likely to link with a newer binary of GLFW that is adequate for your version of Visual Studio.
#if defined(_MSC_VER) && (_MSC_VER >= 1900) && !defined(IMGUI_DISABLE_WIN32_FUNCTIONS)
#pragma comment(lib, "legacy_stdio_definitions")
#endif


class Gui {
    public:
        Gui(std::string window_name, CNN *cnn, Dataset *dataset);
        ~Gui();

        void run();
        bool is_enabled() const {return enabled;};

    private:
        void update();
        static void train_cnn(CNN *cnn, Dataset *dataset, int epochs);
        static void validate_cnn(CNN *cnn, Dataset *dataset);

        std::string window_name;
        bool enabled = true;
        GLFWwindow* window;
        ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
        ImGuiIO io;
        CNN *cnn;
        Dataset *dataset;
        std::thread training_thread;
        std::thread validating_thread;

        bool show_another_window = false;
        bool model_trained = false;
        bool training = false;
        bool validating = false;
        bool join_train_thread = false;
        bool join_validate_thread = false;
        int run_epochs = 1;
        int image_counter = 0;
};


#endif
