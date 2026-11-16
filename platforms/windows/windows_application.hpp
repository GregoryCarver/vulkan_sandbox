#pragma once
#include "windows_platform.hpp"
#include <sandbox/viewer/viewer.hpp>

namespace sandbox {
// Owns one desktop application session. The entry point handles exceptions;
// this type handles configuration, the event loop, and ordered cleanup.
class WindowsApplication {
public:
    static int run(int argc, char** argv);

private:
    struct Options {
        ViewerOptions viewer;
        ViewState initial_state;
        std::filesystem::path assets;
        unsigned frame_limit = 0;
        unsigned fps = 60;
        bool hidden = false;
        bool smoke_test = false;
        bool help = false;
    };
    struct Result {
        unsigned rendered_frames = 0;
        unsigned validation_errors = 0;
    };

    static Options parse_options(int argc, char** argv);
    WindowsApplication(Options options, Result& result);
    ~WindowsApplication() noexcept;
    WindowsApplication(const WindowsApplication&) = delete;
    WindowsApplication& operator=(const WindowsApplication&) = delete;
    void run_loop();
    void smoke_step(unsigned tick);

    Options options_;
    Result& result_; // Owned by run(), which reads it after this session ends.
    // Declaration order ensures the renderer dies before its state and window.
    ViewState state_;
    WindowsPlatform platform_;
    Viewer viewer_;
};
}
