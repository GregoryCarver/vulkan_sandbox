#pragma once

#include <sandbox/core/device.hpp>
#include <sandbox/viewer/input.hpp>
#include <memory>

namespace sandbox {
class Allocator;
class UploadQueue;

// Example scene only: camera, cube geometry, materials and debug UI do not live
// in the reusable Vulkan core. The owner waits a frame-slot fence before prepare.
class Scene {
public:
    Scene(Device& device, Allocator& allocator, UploadQueue& uploads,
          Platform& platform, ViewState& state, uint32_t frames_in_flight);
    ~Scene();
    Scene(const Scene&) = delete;
    Scene& operator=(const Scene&) = delete;

    // Call with the GPU idle, before prepare. UI context survives this operation.
    void on_swapchain_changed(VkFormat color_format, VkFormat depth_format,
                              uint32_t image_count);
    void prepare(uint32_t frame_index, const InputState& input,
                 float delta_seconds, VkExtent2D extent);
    // Called inside the owner's dynamic-rendering color/depth scope.
    void record(VkCommandBuffer command, uint32_t frame_index, VkExtent2D extent,
                VkFormat color_format, VkFormat depth_format, uint32_t image_count);
    // UI uses native framebuffer coordinates in a separate color-only rendering
    // scope, so resolution scaling does not blur text or alter input hit testing.
    void record_ui(VkCommandBuffer command);
    bool wants_text_input() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
