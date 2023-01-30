#include "include/Gui.h"


static void glfw_error_callback(int error, const char* description)
{
    fprintf(stderr, "Glfw Error %d: %s\n", error, description);
}

// Function turn a cv::Mat into a texture, and return the texture ID as a GLuint for use
static GLuint matToTexture(const cv::Mat& mat, GLenum minFilter, GLenum magFilter, GLenum wrapFilter) {
	// Generate a number for our textureID's unique handle
	GLuint textureID;
	glGenTextures(1, &textureID);

	// Bind to our texture handle
	glBindTexture(GL_TEXTURE_2D, textureID);

	// Catch silly-mistake texture interpolation method for magnification
	if (magFilter == GL_LINEAR_MIPMAP_LINEAR ||
		magFilter == GL_LINEAR_MIPMAP_NEAREST ||
		magFilter == GL_NEAREST_MIPMAP_LINEAR ||
		magFilter == GL_NEAREST_MIPMAP_NEAREST)
	{
		std::cout << "You can't use MIPMAPs for magnification - setting filter to GL_LINEAR" << std::endl;
		magFilter = GL_LINEAR;
	}

	// Set texture interpolation methods for minification and magnification
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, minFilter);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, magFilter);

	// Set texture clamping method
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrapFilter);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrapFilter);

	// Set incoming texture format to:
	// GL_BGR       for CV_CAP_OPENNI_BGR_IMAGE,
	// GL_LUMINANCE for CV_CAP_OPENNI_DISPARITY_MAP,
	// Work out other mappings as required ( there's a list in comments in main() )
	GLenum inputColourFormat = 0x80E0;
	// if (mat.channels() == 1)
	// {
	// 	inputColourFormat = GL_LUMINANCE;
	// }

	// Create the texture
	glTexImage2D(GL_TEXTURE_2D,     // Type of texture
		0,                 // Pyramid level (for mip-mapping) - 0 is the top level
		GL_RGB,            // Internal colour format to convert to
		mat.cols,          // Image width  i.e. 640 for Kinect in standard mode
		mat.rows,          // Image height i.e. 480 for Kinect in standard mode
		0,                 // Border width in pixels (can either be 1 or 0)
		inputColourFormat, // Input image format (i.e. GL_RGB, GL_RGBA, GL_BGR etc.)
		GL_UNSIGNED_BYTE,  // Image data type
		mat.ptr());        // The actual image data itself

    // If we're using mipmaps then generate them. Note: This requires OpenGL 3.0 or higher
	if (minFilter == GL_LINEAR_MIPMAP_LINEAR ||
		minFilter == GL_LINEAR_MIPMAP_NEAREST ||
		minFilter == GL_NEAREST_MIPMAP_LINEAR ||
		minFilter == GL_NEAREST_MIPMAP_NEAREST)
	{
		glGenerateMipmap(GL_TEXTURE_2D);
	}

	return textureID;
}

Gui::Gui(std::string window_name, CNN *cnn, Dataset *dataset) {
    // Setup window
    this->cnn = cnn;
    this->dataset = dataset;
    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit()) {
        this->enabled = false;
        return;
    }

    // Decide GL+GLSL versions
    #if defined(IMGUI_IMPL_OPENGL_ES2)
        // GL ES 2.0 + GLSL 100
        const char* glsl_version = "#version 100";
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
        glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_ES_API);
    #elif defined(__APPLE__)
        // GL 3.2 + GLSL 150
        const char* glsl_version = "#version 150";
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
        glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // Required on Mac
    #else
        // GL 3.0 + GLSL 130
        const char* glsl_version = "#version 130";
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    #endif

    // Create window with graphics context
    window = glfwCreateWindow(1280, 720, window_name.c_str(), NULL, NULL);
    if (window == NULL) {
        this->enabled = false;
        return;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable v-stink

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;       // Enable Keyboard Controls
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;           // Enable Docking
    io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;         // Enable Multi-Viewport / Platform Windows
    //io.ConfigViewportsNoAutoMerge = true;

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();

    // When viewports are enabled we tweak WindowRounding/WindowBg so platform windows can look identical to regular ones.
    ImGuiStyle& style = ImGui::GetStyle();
    if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
        style.WindowRounding = 0.0f;
        style.Colors[ImGuiCol_WindowBg].w = 1.0f;
    }

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);
}

void Gui::run() {
    while (!glfwWindowShouldClose(window)) {
        // Poll and handle events (inputs, window resize, etc.)
        // You can read the io.WantCaptureMouse, io.WantCaptureKeyboard flags to tell if dear imgui wants to use your inputs.
        // - When io.WantCaptureMouse is true, do not dispatch mouse input data to your main application, or clear/overwrite your copy of the mouse data.
        // - When io.WantCaptureKeyboard is true, do not dispatch keyboard input data to your main application, or clear/overwrite your copy of the keyboard data.
        // Generally you may always pass all inputs to dear imgui, and hide them from your application based on those two flags.
        glfwPollEvents();

        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        this->update();

        // Rendering
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(clear_color.x * clear_color.w, clear_color.y * clear_color.w, clear_color.z * clear_color.w, clear_color.w);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        // Update and Render additional Platform Windows
        // (Platform functions may change the current OpenGL context, so we save/restore it to make it easier to paste this code elsewhere.
        //  For this specific demo app we could also call glfwMakeContextCurrent(window) directly)
        if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
            GLFWwindow* backup_current_context = glfwGetCurrentContext();
            ImGui::UpdatePlatformWindows();
            ImGui::RenderPlatformWindowsDefault();
            glfwMakeContextCurrent(backup_current_context);
        }

        glfwSwapBuffers(window);
    }
}

Gui::~Gui() {
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(this->window);
    glfwTerminate();
    this->cnn->quit();
    if (this->join_train_thread) {
        this->training_thread.join();
    }
    if (this->join_validate_thread) {
        this->validating_thread.join();
    }
}

void Gui::validate_cnn(CNN *cnn, Dataset *dataset) {
    cnn->validate(*dataset->get_test_set(), *dataset->get_test_labels());
}

void Gui::train_cnn(CNN *cnn, Dataset *dataset, int epochs) {
    cnn->train(*dataset->get_training_set(), *dataset->get_training_labels(), 0.001, epochs);
}

void Gui::update() {
    // 2. Show a simple window that we create ourselves. We use a Begin/End pair to create a named window.
    ImGui::Begin("Setup");                          // Create a window called "Hello, world!" and append into it.
    ImGui::SliderInt("epochs", &run_epochs, 1, 10);
    ImGui::Text("Train the model");               // Display some text (you can use a format strings too)
    if (ImGui::Button("Train")) {
        if (!this->training && !this->cnn->is_trained()) {
            this->training = true;
            this->join_train_thread = true;
            this->training_thread = std::thread(train_cnn, this->cnn, this->dataset, run_epochs);
        }
        else {
            std::cout << "already training" << std::endl;
        }
    }

    if (this->training) {
        ImGui::Text("Training model... %.2f%% trained", this->cnn->get_training_percentage());
    }

    if (this->cnn->is_trained()) {
        ImGui::Text("Model has been trained");
        if (this->training_thread.joinable() && this->join_train_thread) {
            std::cout << "training thread joined" << std::endl;
            this->training_thread.join();
            this->training = false;
            this->join_train_thread = false;
        }
    }

    if (this->cnn->is_trained()) {
        if (ImGui::Button("Validate")) {
            if (!this->validating && !this->cnn->is_validated()) {
                this->validating = true;
                this->join_validate_thread = true;
                this->validating_thread = std::thread(validate_cnn, this->cnn, this->dataset);
            }
            else {
                std::cout << "already validating" << std::endl;
            }
        }
    }

    if (this->cnn->is_validated() && this->join_validate_thread) {
        if (this->validating_thread.joinable() && this->join_validate_thread) {
            std::cout << "validating thread joined" << std::endl;
            this->validating_thread.join();
            this->join_validate_thread = false;
        }
    }

    if (this->cnn->is_validated()) {
        ImGui::Text("Classified %d images correctly (%.2f%%)", cnn->images_correct(), ((float) cnn->images_correct() / (float) this->dataset->get_test_set()->size()) * 100.0);
    }
    ImGui::Checkbox("Preview dataset images", &show_another_window);
	ImGui::End();

    //3. Show another simple window.
    if (show_another_window) {
        Image *im = this->dataset->get_image(this->image_counter);
        std::string label = this->dataset->get_class(im->get_class());
        const cv::Mat image = im->array_to_cv_mat();
        GLuint image_texture = matToTexture(image, GL_LINEAR_MIPMAP_LINEAR, GL_LINEAR, 1);

        ImGui::Begin("Image preview");
        float ims = this->scale;
        ImVec2 imageSize = ImVec2(ims * image.size().width, ims * image.size().height);

        ImGui::Text("texture size = %.0f x %.0f", imageSize.x, imageSize.y);
        ImGui::Text("image size = %d x %d", image.size().width, image.size().height);
        ImGui::SliderFloat("Scale", &this->scale, 1.0, 10.0f);
        ImGui::Image((void*)(intptr_t)image_texture, imageSize);
        ImGui::Text("Image class: %s", label.c_str());

        if (this->cnn->is_trained()) {
            ImGui::Text("Network thinks: %s", this->dataset->get_class(this->cnn->classify(*im)).c_str());
        }
        else {
            ImGui::Text("Train the network to see \nthe network's classification");
        }

        if (ImGui::Button("Next")) {
            if (this->image_counter < (int) this->dataset->get_buffer()->size()) {
                this->image_counter++;
            }
        }

        ImGui::End();
    }
}
