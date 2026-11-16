#include <sandbox/core/selection.hpp>
#include <array>
#include <iostream>
#include <stdexcept>

namespace {
void expect(bool condition, const char* message) { if (!condition) throw std::runtime_error(message); }
}
int main() {
    try {
        using namespace sandbox;
        std::array<QueueSupport, 3> split{{{VK_QUEUE_GRAPHICS_BIT, false}, {VK_QUEUE_TRANSFER_BIT, true}, {VK_QUEUE_GRAPHICS_BIT, true}}};
        auto selected = select_queues(split);
        expect(selected && selected->graphics == 2 && selected->present == 2, "Prefer a combined queue even when graphics-only comes first");
        selected = select_queues(std::span(split).first(2));
        expect(selected && selected->graphics == 0 && selected->present == 1, "Handle distinct graphics/present queues");
        expect(!select_queues(std::span(split).first(1)), "Reject missing presentation support");
        std::array<QueueSupport, 1> empty{{{VK_QUEUE_GRAPHICS_BIT, true, 0}}};
        expect(!select_queues(empty), "Reject zero queue count");
        std::array<VkSurfaceFormatKHR, 1> unusual{{{VK_FORMAT_R8G8B8A8_UNORM, VK_COLOR_SPACE_DISPLAY_P3_NONLINEAR_EXT}}};
        expect(select_surface_format(unusual).colorSpace == VK_COLOR_SPACE_DISPLAY_P3_NONLINEAR_EXT, "Preserve fallback format/colorspace pair");
        expect(select_composite_alpha(VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR) == VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR, "Use supported alpha mode");
        VkSurfaceCapabilitiesKHR caps{};
        caps.minImageCount = 2; caps.maxImageCount = 2;
        expect(select_image_count(caps) == 2, "Respect maximum image count");
        caps.maxImageCount = 0;
        expect(select_image_count(caps) == 3, "Zero maximum means unbounded");
        caps.currentExtent = {800, 600};
        expect(select_extent(caps, {0, 0}).width == 0, "Minimized surface suspends rendering");
        caps.currentExtent = {UINT32_MAX, UINT32_MAX};
        caps.minImageExtent = {64, 64}; caps.maxImageExtent = {4096, 4096};
        auto extent = select_extent(caps, {30, 5000});
        expect(extent.width == 64 && extent.height == 4096, "Clamp undefined surface extent");
        std::cout << "Selection regression checks passed\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
