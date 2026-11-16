#pragma once
#include <vulkan/vulkan.h>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>

namespace sandbox {
struct QueueSupport { VkQueueFlags flags; bool present; uint32_t count = 1; };
struct QueueSelection { uint32_t graphics; uint32_t present; };
inline std::optional<QueueSelection> select_queues(std::span<const QueueSupport> queues) {
    std::optional<uint32_t> graphics, present;
    for (uint32_t i = 0; i < queues.size(); ++i) {
        if (!queues[i].count) continue;
        const bool can_draw = (queues[i].flags & VK_QUEUE_GRAPHICS_BIT) != 0;
        if (can_draw && queues[i].present) return QueueSelection{i, i};
        if (can_draw && !graphics) graphics = i;
        if (queues[i].present && !present) present = i;
    }
    if (graphics && present) return QueueSelection{*graphics, *present};
    return std::nullopt;
}
inline VkSurfaceFormatKHR select_surface_format(std::span<const VkSurfaceFormatKHR> formats) {
    if (formats.empty()) throw std::runtime_error("Surface exposes no formats");
    if (formats.size() == 1 && formats[0].format == VK_FORMAT_UNDEFINED)
        return {VK_FORMAT_B8G8R8A8_SRGB, formats[0].colorSpace};
    for (auto format : formats)
        if ((format.format == VK_FORMAT_B8G8R8A8_SRGB || format.format == VK_FORMAT_R8G8B8A8_SRGB)
            && format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) return format;
    return formats.front();
}
inline VkCompositeAlphaFlagBitsKHR select_composite_alpha(VkCompositeAlphaFlagsKHR supported) {
    for (auto flag : {VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR, VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR,
                     VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR, VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR})
        if (supported & flag) return flag;
    throw std::runtime_error("Surface exposes no composite alpha mode");
}
inline uint32_t select_image_count(const VkSurfaceCapabilitiesKHR& caps) {
    auto count = std::max(2u, caps.minImageCount + 1);
    return caps.maxImageCount ? std::min(count, caps.maxImageCount) : count;
}
inline VkExtent2D select_extent(const VkSurfaceCapabilitiesKHR& caps, VkExtent2D requested) {
    if (!requested.width || !requested.height) return {};
    if (caps.currentExtent.width != std::numeric_limits<uint32_t>::max()) return caps.currentExtent;
    return {std::clamp(requested.width, caps.minImageExtent.width, caps.maxImageExtent.width),
            std::clamp(requested.height, caps.minImageExtent.height, caps.maxImageExtent.height)};
}
}
