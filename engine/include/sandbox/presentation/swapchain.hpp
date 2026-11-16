#pragma once
#include <sandbox/core/device.hpp>
#include <vector>

namespace sandbox {
// The owner must stop GPU work before recreation/destruction. Frame fences and
// acquire semaphores are per frame; presentation semaphores belong to images.
class Swapchain {
public:
    Swapchain(Device& device, Platform& platform, bool vsync);
    ~Swapchain();
    Swapchain(const Swapchain&) = delete;
    Swapchain& operator=(const Swapchain&) = delete;
    bool recreate();
    VkSwapchainKHR handle() const { return current_.handle; }
    VkExtent2D extent() const { return extent_; }
    VkFormat format() const { return format_; }
    uint32_t image_count() const { return static_cast<uint32_t>(current_.images.size()); }
    VkImage image(uint32_t i) const { return current_.images.at(i); }
    VkImageView view(uint32_t i) const { return current_.views.at(i); }
    VkSemaphore present_semaphore(uint32_t i) const { return current_.present_semaphores.at(i); }
    bool can_blit() const { return can_blit_; }
    bool has_retired() const { return !retired_.empty(); }
    // Call only after the acquisition fence signaled. Reacquiring an image
    // presented by this generation proves previous generations have retired.
    void acquired(uint32_t index);
    // Returns a reset fence to chain into VkPresentInfoKHR, or null on drivers
    // without swapchain maintenance1. The same image's previous use is waited.
    VkFence prepare_present(uint32_t index);
    // Must be called immediately after vkQueuePresentKHR, including error paths.
    void presented(uint32_t index, VkResult result);
private:
    struct Generation {
        VkSwapchainKHR handle = VK_NULL_HANDLE;
        std::vector<VkImage> images;
        std::vector<VkImageView> views;
        std::vector<VkSemaphore> present_semaphores;
        std::vector<VkFence> present_fences;
        std::vector<bool> present_pending;
        std::vector<bool> presented;
    };
    void destroy(Generation&) noexcept;
    Device& device_;
    Platform& platform_;
    bool vsync_;
    bool can_blit_ = false;
    VkExtent2D extent_{};
    VkFormat format_ = VK_FORMAT_UNDEFINED;
    Generation current_;
    std::vector<Generation> retired_;
};
}
