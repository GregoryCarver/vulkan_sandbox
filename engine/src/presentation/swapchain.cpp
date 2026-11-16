#include <sandbox/presentation/swapchain.hpp>
#include <sandbox/core/selection.hpp>
#include <algorithm>
#include <array>
#include <utility>

namespace sandbox {
Swapchain::Swapchain(Device& device, Platform& platform, bool vsync)
    : device_(device), platform_(platform), vsync_(vsync) {}
Swapchain::~Swapchain() {
    destroy(current_);
    for (auto& generation : retired_) destroy(generation);
}
void Swapchain::destroy(Generation& generation) noexcept {
    // Present fences cover presentation-engine references, which a graphics
    // submission fence (or vkDeviceWaitIdle alone) does not establish.
    for (size_t i = 0; i < generation.present_fences.size(); ++i) {
        if (generation.present_pending[i])
            (void)vkWaitForFences(device_.logical(), 1, &generation.present_fences[i], VK_TRUE, UINT64_MAX);
        vkDestroyFence(device_.logical(), generation.present_fences[i], nullptr);
    }
    for (auto view : generation.views) vkDestroyImageView(device_.logical(), view, nullptr);
    for (auto semaphore : generation.present_semaphores) vkDestroySemaphore(device_.logical(), semaphore, nullptr);
    if (generation.handle) vkDestroySwapchainKHR(device_.logical(), generation.handle, nullptr);
    generation = {};
}
void Swapchain::acquired(uint32_t index) {
    if (current_.presented.at(index)) {
        for (auto& generation : retired_) destroy(generation);
        retired_.clear();
    }
}
VkFence Swapchain::prepare_present(uint32_t index) {
    if (!device_.present_fences_enabled()) return VK_NULL_HANDLE;
    const VkFence fence = current_.present_fences.at(index);
    if (current_.present_pending.at(index)) {
        check_vk(vkWaitForFences(device_.logical(), 1, &fence, VK_TRUE, UINT64_MAX), "Wait presentation fence");
        current_.present_pending[index] = false;
    }
    check_vk(vkResetFences(device_.logical(), 1, &fence), "Reset presentation fence");
    return fence;
}
void Swapchain::presented(uint32_t index, VkResult result) {
    // These results still enqueue the semaphore wait and presentation operations.
    // Allocation failures leave synchronization objects unchanged, and therefore
    // must not leave an unsignaled fence marked as pending during shutdown.
    const bool enqueued = result == VK_SUCCESS || result == VK_SUBOPTIMAL_KHR ||
        result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_ERROR_SURFACE_LOST_KHR ||
        result == VK_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT;
    if (!enqueued) return;
    current_.presented.at(index) = true;
    if (device_.present_fences_enabled()) current_.present_pending.at(index) = true;
}
bool Swapchain::recreate() {
    auto requested = platform_.framebuffer_extent();
    if (!requested.width || !requested.height) return false;
    VkSurfaceCapabilitiesKHR caps{};
    check_vk(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device_.physical(), device_.surface(), &caps), "Query surface capabilities");
    auto extent = select_extent(caps, requested);
    if (!extent.width || !extent.height) return false;
    // Keep camera, touch, and UI coordinates in the same upright space. A
    // pre-rotated rendering path must rotate all three together before it can
    // replace this compositor-managed identity path.
    if (!(caps.supportedTransforms & VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR))
        throw std::runtime_error("This sample requires identity surface transform; pre-rotated surfaces are not supported yet");
    uint32_t count = 0;
    check_vk(vkGetPhysicalDeviceSurfaceFormatsKHR(device_.physical(), device_.surface(), &count, nullptr), "Count surface formats");
    std::vector<VkSurfaceFormatKHR> formats(count);
    check_vk(vkGetPhysicalDeviceSurfaceFormatsKHR(device_.physical(), device_.surface(), &count, formats.data()), "Read surface formats");
    auto selected = select_surface_format(formats);
    check_vk(vkGetPhysicalDeviceSurfacePresentModesKHR(device_.physical(), device_.surface(), &count, nullptr), "Count present modes");
    std::vector<VkPresentModeKHR> modes(count);
    check_vk(vkGetPhysicalDeviceSurfacePresentModesKHR(device_.physical(), device_.surface(), &count, modes.data()), "Read present modes");
    VkPresentModeKHR present = VK_PRESENT_MODE_FIFO_KHR;
    if (!vsync_) {
        if (std::ranges::find(modes, VK_PRESENT_MODE_MAILBOX_KHR) != modes.end()) present = VK_PRESENT_MODE_MAILBOX_KHR;
        else if (std::ranges::find(modes, VK_PRESENT_MODE_IMMEDIATE_KHR) != modes.end()) present = VK_PRESENT_MODE_IMMEDIATE_KHR;
    }
    if (!(caps.supportedUsageFlags & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT))
        throw std::runtime_error("Surface images cannot be color attachments");
    VkFormatProperties properties{};
    vkGetPhysicalDeviceFormatProperties(device_.physical(), selected.format, &properties);
    const bool blit = (caps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_DST_BIT)
        && (properties.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_SRC_BIT)
        && (properties.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_DST_BIT)
        && (properties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT);
    std::array<uint32_t, 2> families{device_.graphics_family(), device_.present_family()};
    VkSwapchainCreateInfoKHR create{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
    create.surface = device_.surface();
    create.minImageCount = select_image_count(caps);
    create.imageFormat = selected.format;
    create.imageColorSpace = selected.colorSpace;
    create.imageExtent = extent;
    create.imageArrayLayers = 1;
    create.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | (blit ? VK_IMAGE_USAGE_TRANSFER_DST_BIT : 0);
    create.imageSharingMode = families[0] == families[1] ? VK_SHARING_MODE_EXCLUSIVE : VK_SHARING_MODE_CONCURRENT;
    create.queueFamilyIndexCount = families[0] == families[1] ? 0u : 2u;
    create.pQueueFamilyIndices = families.data();
    // Keep sample/input coordinates upright. Android's compositor can rotate
    // identity presentation; pre-rotation is an optional later optimization.
    create.preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
    create.compositeAlpha = select_composite_alpha(caps.supportedCompositeAlpha);
    create.presentMode = present;
    create.clipped = VK_TRUE;
    create.oldSwapchain = current_.handle;
    Generation next;
    retired_.reserve(retired_.size() + 1);
    try {
        check_vk(vkCreateSwapchainKHR(device_.logical(), &create, nullptr, &next.handle), "Create swapchain");
        check_vk(vkGetSwapchainImagesKHR(device_.logical(), next.handle, &count, nullptr), "Count swapchain images");
        next.images.resize(count);
        check_vk(vkGetSwapchainImagesKHR(device_.logical(), next.handle, &count, next.images.data()), "Read swapchain images");
        next.presented.resize(count, false);
        if (device_.present_fences_enabled()) {
            next.present_fences.reserve(count);
            next.present_pending.resize(count, false);
        }
        next.views.reserve(count);
        next.present_semaphores.reserve(count);
        for (auto image : next.images) {
            VkImageViewCreateInfo view{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            view.image = image;
            view.viewType = VK_IMAGE_VIEW_TYPE_2D;
            view.format = selected.format;
            view.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            VkImageView handle{};
            check_vk(vkCreateImageView(device_.logical(), &view, nullptr, &handle), "Create swapchain view");
            next.views.push_back(handle);
            VkSemaphoreCreateInfo semaphore{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
            VkSemaphore signal{};
            check_vk(vkCreateSemaphore(device_.logical(), &semaphore, nullptr, &signal), "Create presentation semaphore");
            next.present_semaphores.push_back(signal);
            if (device_.present_fences_enabled()) {
                VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
                VkFence fence{};
                check_vk(vkCreateFence(device_.logical(), &fence_info, nullptr, &fence), "Create presentation fence");
                next.present_fences.push_back(fence);
            }
        }
    } catch (...) { destroy(next); throw; }
    if (current_.handle) {
        if (device_.present_fences_enabled()) destroy(current_);
        else retired_.push_back(std::move(current_));
    }
    current_ = std::move(next);
    extent_ = extent;
    format_ = selected.format;
    can_blit_ = blit;
    return true;
}
}
