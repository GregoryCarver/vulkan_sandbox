#pragma once
#include <sandbox/core/platform.hpp>
#include <sandbox/viewer/input.hpp>
#include <filesystem>
struct GLFWwindow;

namespace sandbox {
class WindowsPlatform final : public Platform {
public:
    WindowsPlatform(int width, int height, bool hidden, std::filesystem::path asset_override = {});
    ~WindowsPlatform() override;
    WindowsPlatform(const WindowsPlatform&) = delete;
    WindowsPlatform& operator=(const WindowsPlatform&) = delete;
    std::vector<const char*> instance_extensions() const override;
    VkSurfaceKHR create_surface(VkInstance instance) const override;
    VkExtent2D framebuffer_extent() const override;
    float content_scale() const override;
    std::vector<std::byte> read_asset(std::string_view path) const override;
    std::string writable_path(std::string_view filename) const override;
    InputState poll_input();
    bool take_resize();
    bool should_close() const;
    GLFWwindow* window() const { return window_; }
private:
    GLFWwindow* window_ = nullptr;
    std::filesystem::path assets_;
    InputState input_;
    bool resized_ = false;
    double previous_x_ = 0, previous_y_ = 0;
    bool was_looking_ = false;
};
}
