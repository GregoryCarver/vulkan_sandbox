#include "windows_application.hpp"
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

namespace sandbox {
WindowsApplication::Options WindowsApplication::parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string argument(argv[i]);
        auto value = [&]() -> std::string {
            if (++i >= argc) throw std::runtime_error("Missing value for " + argument);
            return argv[i];
        };
        if (argument == "--frames") options.frame_limit = static_cast<unsigned>(std::stoul(value()));
        else if (argument == "--fps") options.fps = static_cast<unsigned>(std::stoul(value()));
        else if (argument == "--render-scale") options.viewer.render_scale = std::stof(value());
        else if (argument == "--assets") options.assets = value();
        else if (argument == "--hidden") options.hidden = true;
        else if (argument == "--no-ui") options.initial_state.show_ui = false;
        else if (argument == "--no-validation") options.viewer.validation = false;
        else if (argument == "--no-vsync") options.viewer.vsync = false;
        else if (argument == "--legacy-present") options.viewer.present_fences = false;
        else if (argument == "--stress") options.initial_state.stream_mesh_updates = true;
        else if (argument == "--smoke-test") {
            options.smoke_test = true;
            if (!options.frame_limit) options.frame_limit = 180;
        } else if (argument == "--help") {
            options.help = true;
            return options;
        } else throw std::runtime_error("Unknown argument: " + argument);
    }
    if (options.fps > 1000) throw std::runtime_error("FPS must be 0 (uncapped) or at most 1000");
    return options;
}

int WindowsApplication::run(int argc, char** argv) {
    auto options = parse_options(argc, argv);
    if (options.help) {
        std::cout << "sandbox_windows [--frames N] [--fps N] [--render-scale 0.25..1]\n"
            "  [--assets DIR] [--hidden] [--no-ui] [--no-validation] [--no-vsync]\n"
            "  [--stress] [--smoke-test] [--legacy-present]\nWASD / Q,E: move. Right drag: look. F1: UI.\n";
        return 0;
    }
    Result result;
    {
        WindowsApplication application(std::move(options), result);
        application.run_loop();
    }
    // Read the result after RAII cleanup so teardown errors affect the exit code.
    std::cout << "Rendered " << result.rendered_frames << " frames; validation errors: "
              << result.validation_errors << '\n';
    return result.validation_errors ? 2 : 0;
}

WindowsApplication::WindowsApplication(Options options, Result& result)
    : options_(std::move(options)), result_(result), state_(options_.initial_state),
      platform_(1280, 800, options_.hidden, options_.assets), viewer_(platform_, state_, options_.viewer) {}

WindowsApplication::~WindowsApplication() noexcept {
    // Shutdown releases GPU resources before the window dies and captures
    // validation messages emitted during Vulkan destruction.
    result_.validation_errors = viewer_.shutdown();
}

void WindowsApplication::smoke_step(unsigned tick) {
    if (tick == 30) glfwSetWindowSize(platform_.window(), 960, 540);
    if (tick == 60) glfwSetWindowSize(platform_.window(), 1024, 768);
    if (tick == 90) glfwIconifyWindow(platform_.window());
    if (tick == 100) {
        glfwRestoreWindow(platform_.window());
        if (options_.hidden) glfwHideWindow(platform_.window());
    }
    if (tick == 130) glfwSetWindowSize(platform_.window(), 1280, 800);
}

void WindowsApplication::run_loop() {
    using Clock = std::chrono::steady_clock;
    auto previous = Clock::now();
    unsigned ticks = 0;
    while (!platform_.should_close() && (!options_.frame_limit || result_.rendered_frames < options_.frame_limit)) {
        const auto start = Clock::now();
        auto input = platform_.poll_input();
        if (options_.smoke_test) smoke_step(ticks);
        if (platform_.take_resize()) viewer_.request_resize();
        const float delta = std::chrono::duration<float>(start - previous).count();
        previous = start;
        if (viewer_.draw(input, delta)) ++result_.rendered_frames;
        else glfwWaitEventsTimeout(0.01);
        ++ticks;
        if (options_.smoke_test && ticks > std::max(600u, options_.frame_limit * 4u))
            throw std::runtime_error("Smoke test timed out without presenting enough frames");
        if (options_.fps) std::this_thread::sleep_until(start + std::chrono::microseconds(1000000 / options_.fps));
    }
    // Keep device-loss reporting on the throwing path; destructors cannot report it.
    viewer_.wait_idle();
}
}
