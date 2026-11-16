#pragma once
#include <sandbox/core/platform.hpp>
#include <sandbox/viewer/input.hpp>
#include <memory>

namespace sandbox {
struct ViewerOptions {
    bool validation = true;
    bool vsync = true;
    bool present_fences = true;
    // Scale is a quality setting; presentation still uses the native surface size.
    float render_scale = 1.0f;
};
class Viewer {
public:
    Viewer(Platform& platform, ViewState& state, ViewerOptions options = {});
    ~Viewer();
    Viewer(const Viewer&) = delete;
    Viewer& operator=(const Viewer&) = delete;
    // False means presentation was skipped (zero extent or recreation), not failure.
    bool draw(const InputState& input, float delta_seconds);
    void request_resize();
    void wait_idle();
    unsigned validation_errors() const;
    // Releases all GPU objects and reports validation errors including teardown.
    // No other method may be called after shutdown.
    unsigned shutdown();
    bool wants_text_input() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
