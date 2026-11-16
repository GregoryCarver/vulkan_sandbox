#pragma once

#include <sandbox/core/platform.hpp>
#include <atomic>
#include <cstdint>
#include <string>
#include <memory>

namespace sandbox {
void check_vk(VkResult result, const char* operation);

class Device {
public:
    explicit Device(Platform& platform, bool validation = true, bool present_fences = true);
    ~Device();
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;

    VkInstance instance() const { return instance_; }
    VkPhysicalDevice physical() const { return physical_; }
    VkDevice logical() const { return device_; }
    VkSurfaceKHR surface() const { return surface_; }
    VkQueue graphics_queue() const { return graphics_queue_; }
    VkQueue present_queue() const { return present_queue_; }
    uint32_t graphics_family() const { return graphics_family_; }
    uint32_t present_family() const { return present_family_; }
    const VkPhysicalDeviceProperties& properties() const { return properties_; }
    bool validation_enabled() const { return validation_enabled_; }
    bool present_fences_enabled() const { return present_fences_enabled_; }
    uint32_t validation_errors() const { return validation_errors_->load(); }
    std::shared_ptr<std::atomic<uint32_t>> validation_counter() const { return validation_errors_; }
    void wait_idle() const;

private:
    void destroy() noexcept;
    static VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(VkDebugUtilsMessageSeverityFlagBitsEXT,
        VkDebugUtilsMessageTypeFlagsEXT, const VkDebugUtilsMessengerCallbackDataEXT*, void*);
    VkInstance instance_ = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT messenger_ = VK_NULL_HANDLE;
    VkSurfaceKHR surface_ = VK_NULL_HANDLE;
    VkPhysicalDevice physical_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue graphics_queue_ = VK_NULL_HANDLE;
    VkQueue present_queue_ = VK_NULL_HANDLE;
    uint32_t graphics_family_ = 0;
    uint32_t present_family_ = 0;
    VkPhysicalDeviceProperties properties_{};
    bool validation_enabled_ = false;
    bool present_fences_enabled_ = false;
    std::shared_ptr<std::atomic<uint32_t>> validation_errors_ = std::make_shared<std::atomic<uint32_t>>(0);
};
}
