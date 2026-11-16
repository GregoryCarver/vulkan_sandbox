#pragma once

#include <vulkan/vulkan.h>
#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace sandbox {
// The host owns its native window. Returned Vulkan surfaces belong to Device.
class Platform {
public:
    virtual ~Platform() = default;
    virtual std::vector<const char*> instance_extensions() const = 0;
    virtual VkSurfaceKHR create_surface(VkInstance instance) const = 0;
    virtual VkExtent2D framebuffer_extent() const = 0;
    virtual float content_scale() const { return 1.0f; }
    virtual std::vector<std::byte> read_asset(std::string_view path) const = 0;
    virtual std::string writable_path(std::string_view filename) const = 0;
};
}
