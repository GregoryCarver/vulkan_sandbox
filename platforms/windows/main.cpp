#include "windows_platform.hpp"
#include <sandbox/viewer/viewer.hpp>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

int main(int argc, char** argv) {
    try {
        sandbox::ViewerOptions options;
        sandbox::ViewState state;
        unsigned limit = 0, fps = 60;
        bool hidden = false, smoke = false;
        std::filesystem::path assets;
        for (int i = 1; i < argc; ++i) {
            const std::string argument(argv[i]);
            auto value = [&]() -> std::string { if (++i >= argc) throw std::runtime_error("Missing value for " + argument); return argv[i]; };
            if (argument == "--frames") limit = static_cast<unsigned>(std::stoul(value()));
            else if (argument == "--fps") fps = static_cast<unsigned>(std::stoul(value()));
            else if (argument == "--render-scale") options.render_scale = std::stof(value());
            else if (argument == "--assets") assets = value();
            else if (argument == "--hidden") hidden = true;
            else if (argument == "--no-ui") state.show_ui = false;
            else if (argument == "--no-validation") options.validation = false;
            else if (argument == "--no-vsync") options.vsync = false;
            else if (argument == "--legacy-present") options.present_fences = false;
            else if (argument == "--stress") state.stream_mesh_updates = true;
            else if (argument == "--smoke-test") { smoke = true; if (!limit) limit = 180; }
            else if (argument == "--help") {
                std::cout << "sandbox_windows [--frames N] [--fps N] [--render-scale 0.25..1]\n"
                    "  [--assets DIR] [--hidden] [--no-ui] [--no-validation] [--no-vsync]\n"
                    "  [--stress] [--smoke-test] [--legacy-present]\nWASD / Q,E: move. Right drag: look. F1: UI.\n";
                return 0;
            } else throw std::runtime_error("Unknown argument: " + argument);
        }
        if (fps > 1000) throw std::runtime_error("FPS must be 0 (uncapped) or at most 1000");
        sandbox::WindowsPlatform platform(1280, 800, hidden, assets);
        sandbox::Viewer viewer(platform, state, options);
        using Clock = std::chrono::steady_clock;
        auto previous = Clock::now();
        unsigned rendered = 0, ticks = 0;
        while (!platform.should_close() && (!limit || rendered < limit)) {
            const auto start = Clock::now();
            auto input = platform.poll_input();
            if (smoke) {
                if (ticks == 30) glfwSetWindowSize(platform.window(), 960, 540);
                if (ticks == 60) glfwSetWindowSize(platform.window(), 1024, 768);
                if (ticks == 90) glfwIconifyWindow(platform.window());
                if (ticks == 100) { glfwRestoreWindow(platform.window()); if (hidden) glfwHideWindow(platform.window()); }
                if (ticks == 130) glfwSetWindowSize(platform.window(), 1280, 800);
            }
            if (platform.take_resize()) viewer.request_resize();
            const float delta = std::chrono::duration<float>(start - previous).count();
            previous = start;
            if (viewer.draw(input, delta)) ++rendered;
            else glfwWaitEventsTimeout(0.01);
            ++ticks;
            if (smoke && ticks > std::max(600u, limit * 4u))
                throw std::runtime_error("Smoke test timed out without presenting enough frames");
            if (fps) std::this_thread::sleep_until(start + std::chrono::microseconds(1000000 / fps));
        }
        viewer.wait_idle();
        const auto errors = viewer.shutdown();
        std::cout << "Rendered " << rendered << " frames; validation errors: " << errors << '\n';
        return errors ? 2 : 0;
    } catch (const std::exception& error) {
        std::cerr << "Sandbox startup/rendering failed: " << error.what() << '\n';
        return 1;
    }
}
