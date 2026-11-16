#pragma once
#include <sandbox/core/platform.hpp>
#include <game-activity/native_app_glue/android_native_app_glue.h>

namespace sandbox::android {
class AndroidPlatform final : public Platform {
public:
    explicit AndroidPlatform(android_app& app) : app_(app) {}
    std::vector<const char*> instance_extensions() const override;
    VkSurfaceKHR create_surface(VkInstance instance) const override;
    VkExtent2D framebuffer_extent() const override;
    float content_scale() const override;
    std::vector<std::byte> read_asset(std::string_view path) const override;
    std::string writable_path(std::string_view filename) const override;
private:
    android_app& app_;
};
}
