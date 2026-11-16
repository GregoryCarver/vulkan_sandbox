#include <sandbox/viewer/viewer.hpp>
#include <sandbox/viewer/scene.hpp>
#include <sandbox/gpu/resources.hpp>
#include <sandbox/presentation/swapchain.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace sandbox {
namespace {
constexpr uint32_t frame_count = 2;
void barrier(VkCommandBuffer command, VkImage image, VkImageAspectFlags aspect,
    VkImageLayout before, VkImageLayout after, VkPipelineStageFlags2 source_stage,
    VkAccessFlags2 source_access, VkPipelineStageFlags2 destination_stage, VkAccessFlags2 destination_access) {
    VkImageMemoryBarrier2 image_barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    image_barrier.srcStageMask = source_stage;
    image_barrier.srcAccessMask = source_access;
    image_barrier.dstStageMask = destination_stage;
    image_barrier.dstAccessMask = destination_access;
    image_barrier.oldLayout = before;
    image_barrier.newLayout = after;
    image_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    image_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    image_barrier.image = image;
    image_barrier.subresourceRange = {aspect, 0, 1, 0, 1};
    VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dependency.imageMemoryBarrierCount = 1;
    dependency.pImageMemoryBarriers = &image_barrier;
    vkCmdPipelineBarrier2(command, &dependency);
}
VkFormat depth_format(Device& device) {
    for (auto format : {VK_FORMAT_D32_SFLOAT, VK_FORMAT_D16_UNORM}) {
        VkFormatProperties properties{};
        vkGetPhysicalDeviceFormatProperties(device.physical(), format, &properties);
        if (properties.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) return format;
    }
    throw std::runtime_error("No supported depth attachment format");
}
struct Frame {
    Device& device;
    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBuffer command = VK_NULL_HANDLE;
    VkFence complete = VK_NULL_HANDLE;
    VkFence acquired = VK_NULL_HANDLE;
    VkSemaphore available = VK_NULL_HANDLE;
    Image depth, color;
    explicit Frame(Device& owner) : device(owner) {
        try {
            VkCommandPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
            pool_info.queueFamilyIndex = owner.graphics_family();
            check_vk(vkCreateCommandPool(owner.logical(), &pool_info, nullptr, &pool), "Create frame command pool");
            VkCommandBufferAllocateInfo allocate{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
            allocate.commandPool = pool;
            allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            allocate.commandBufferCount = 1;
            check_vk(vkAllocateCommandBuffers(owner.logical(), &allocate, &command), "Allocate frame command buffer");
            VkFenceCreateInfo fence{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
            fence.flags = VK_FENCE_CREATE_SIGNALED_BIT;
            check_vk(vkCreateFence(owner.logical(), &fence, nullptr, &complete), "Create frame fence");
            fence.flags = 0;
            check_vk(vkCreateFence(owner.logical(), &fence, nullptr, &acquired), "Create acquisition fence");
            VkSemaphoreCreateInfo semaphore{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
            check_vk(vkCreateSemaphore(owner.logical(), &semaphore, nullptr, &available), "Create acquisition semaphore");
        } catch (...) { destroy(); throw; }
    }
    ~Frame() { destroy(); }
    void destroy() noexcept {
        if (available) vkDestroySemaphore(device.logical(), available, nullptr);
        if (acquired) vkDestroyFence(device.logical(), acquired, nullptr);
        if (complete) vkDestroyFence(device.logical(), complete, nullptr);
        if (pool) vkDestroyCommandPool(device.logical(), pool, nullptr);
    }
};
}

struct Viewer::Impl {
    Platform& platform;
    ViewerOptions options;
    Device device;
    Allocator allocator;
    UploadQueue uploads;
    Swapchain swapchain;
    std::array<std::unique_ptr<Frame>, frame_count> frames;
    std::unique_ptr<Scene> scene;
    VkFormat depth;
    VkExtent2D render_extent{};
    uint32_t current_frame = 0;
    bool resize_pending = true;
    bool scaled = false;
    InputState pending_input;
    Impl(Platform& host, ViewState& state, ViewerOptions settings)
        : platform(host), options(settings), device(host, settings.validation, settings.present_fences), allocator(device),
          uploads(device, allocator), swapchain(device, host, settings.vsync), depth(depth_format(device)) {
        if (!std::isfinite(options.render_scale) || options.render_scale < 0.25f || options.render_scale > 1.0f)
            throw std::invalid_argument("Render scale must be in [0.25, 1]");
        for (auto& frame : frames) frame = std::make_unique<Frame>(device);
        scene = std::make_unique<Scene>(device, allocator, uploads, host, state, frame_count);
    }
    ~Impl() {
        // Also runs during exceptional draw exits, before dependent objects die.
        vkDeviceWaitIdle(device.logical());
        scene.reset();
        for (auto& frame : frames) frame.reset();
    }
    bool rebuild() {
        device.wait_idle();
        if (!swapchain.recreate()) return false;
        const auto extent = swapchain.extent();
        scaled = options.render_scale < 1.0f && swapchain.can_blit();
        if (options.render_scale < 1.0f && !scaled)
            std::cerr << "Surface format cannot be blitted; rendering at native resolution\n";
        render_extent = scaled ? VkExtent2D{
            std::max(1u, static_cast<uint32_t>(extent.width * options.render_scale)),
            std::max(1u, static_cast<uint32_t>(extent.height * options.render_scale))} : extent;
        for (auto& frame : frames) {
            frame->depth = Image(allocator, ImageDesc{render_extent, depth,
                VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_IMAGE_ASPECT_DEPTH_BIT});
            frame->color = scaled ? Image(allocator, ImageDesc{render_extent, swapchain.format(),
                VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT}) : Image{};
        }
        scene->on_swapchain_changed(swapchain.format(), depth, swapchain.image_count());
        resize_pending = false;
        std::cout << "Swapchain: " << extent.width << 'x' << extent.height << ", "
                  << swapchain.image_count() << " images, scene " << render_extent.width << 'x' << render_extent.height << '\n';
        return true;
    }
    void record(Frame& frame, uint32_t image_index) {
        auto command = frame.command;
        check_vk(vkResetCommandPool(device.logical(), frame.pool, 0), "Reset frame command pool");
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        check_vk(vkBeginCommandBuffer(command, &begin), "Begin frame");
        auto target = scaled ? frame.color.handle() : swapchain.image(image_index);
        auto view = scaled ? frame.color.view() : swapchain.view(image_index);
        barrier(command, target, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            VK_ACCESS_2_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
        barrier(command, frame.depth.handle(), VK_IMAGE_ASPECT_DEPTH_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            VK_ACCESS_2_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
            VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);
        VkRenderingAttachmentInfo color_attachment{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
        color_attachment.imageView = view;
        color_attachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        color_attachment.clearValue.color = {{0.025f, 0.035f, 0.055f, 1.0f}};
        VkRenderingAttachmentInfo depth_attachment{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
        depth_attachment.imageView = frame.depth.view();
        depth_attachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        depth_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depth_attachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depth_attachment.clearValue.depthStencil = {1.0f, 0};
        VkRenderingInfo rendering{VK_STRUCTURE_TYPE_RENDERING_INFO};
        rendering.renderArea.extent = render_extent;
        rendering.layerCount = 1;
        rendering.colorAttachmentCount = 1;
        rendering.pColorAttachments = &color_attachment;
        rendering.pDepthAttachment = &depth_attachment;
        vkCmdBeginRendering(command, &rendering);
        scene->record(command, current_frame, render_extent, swapchain.format(), depth, swapchain.image_count());
        vkCmdEndRendering(command);
        if (scaled) {
            barrier(command, target, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_2_BLIT_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
            barrier(command, swapchain.image(image_index), VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_2_NONE, 0,
                VK_PIPELINE_STAGE_2_BLIT_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
            VkImageBlit region{};
            region.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            region.dstSubresource = region.srcSubresource;
            region.srcOffsets[1] = {static_cast<int32_t>(render_extent.width), static_cast<int32_t>(render_extent.height), 1};
            region.dstOffsets[1] = {static_cast<int32_t>(swapchain.extent().width), static_cast<int32_t>(swapchain.extent().height), 1};
            vkCmdBlitImage(command, target, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, swapchain.image(image_index),
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region, VK_FILTER_LINEAR);
            barrier(command, swapchain.image(image_index), VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_PIPELINE_STAGE_2_BLIT_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
        } else {
            barrier(command, target, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
        }
        color_attachment.imageView = swapchain.view(image_index);
        color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
        rendering.renderArea.extent = swapchain.extent();
        rendering.pDepthAttachment = nullptr;
        vkCmdBeginRendering(command, &rendering);
        scene->record_ui(command);
        vkCmdEndRendering(command);
        barrier(command, swapchain.image(image_index), VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
            VK_PIPELINE_STAGE_2_NONE, 0);
        check_vk(vkEndCommandBuffer(command), "End frame");
    }
    bool draw(const InputState& input, float delta) {
        // Acquisition can skip a frame. Preserve discrete events until the UI
        // actually consumes them rather than dropping a click during resize.
        InputState merged = input;
        merged.events.insert(merged.events.begin(), pending_input.events.begin(), pending_input.events.end());
        merged.text = pending_input.text + input.text;
        merged.scroll_y += pending_input.scroll_y;
        merged.toggle_ui = input.toggle_ui != pending_input.toggle_ui;
        pending_input = std::move(merged);
        auto size = platform.framebuffer_extent();
        if (!size.width || !size.height) { resize_pending = true; return false; }
        if (resize_pending && !rebuild()) return false;
        auto& frame = *frames[current_frame];
        check_vk(vkWaitForFences(device.logical(), 1, &frame.complete, VK_TRUE, UINT64_MAX), "Wait frame fence");
        uploads.collect();
        const bool track_acquisition = swapchain.has_retired();
        if (track_acquisition) check_vk(vkResetFences(device.logical(), 1, &frame.acquired), "Reset acquisition fence");
        uint32_t image_index = 0;
        auto acquire = vkAcquireNextImageKHR(device.logical(), swapchain.handle(), UINT64_MAX,
            frame.available, track_acquisition ? frame.acquired : VK_NULL_HANDLE, &image_index);
        if (acquire == VK_ERROR_OUT_OF_DATE_KHR) { resize_pending = true; return false; }
        if (acquire != VK_SUCCESS && acquire != VK_SUBOPTIMAL_KHR) check_vk(acquire, "Acquire swapchain image");
        if (track_acquisition) {
            check_vk(vkWaitForFences(device.logical(), 1, &frame.acquired, VK_TRUE, UINT64_MAX), "Wait acquired image");
            swapchain.acquired(image_index);
        }
        scene->prepare(current_frame, pending_input, std::clamp(delta, 0.001f, 0.1f), swapchain.extent());
        pending_input.events.clear();
        pending_input.text.clear();
        pending_input.scroll_y = 0;
        pending_input.toggle_ui = false;
        record(frame, image_index);
        VkSemaphoreSubmitInfo wait{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
        wait.semaphore = frame.available;
        wait.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        VkSemaphoreSubmitInfo signal{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
        signal.semaphore = swapchain.present_semaphore(image_index);
        signal.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        VkCommandBufferSubmitInfo command{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
        command.commandBuffer = frame.command;
        VkSubmitInfo2 submit{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
        submit.waitSemaphoreInfoCount = 1; submit.pWaitSemaphoreInfos = &wait;
        submit.commandBufferInfoCount = 1; submit.pCommandBufferInfos = &command;
        submit.signalSemaphoreInfoCount = 1; submit.pSignalSemaphoreInfos = &signal;
        check_vk(vkResetFences(device.logical(), 1, &frame.complete), "Reset frame fence");
        check_vk(vkQueueSubmit2(device.graphics_queue(), 1, &submit, frame.complete), "Submit frame");
        auto swapchain_handle = swapchain.handle();
        VkPresentInfoKHR present{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
        present.waitSemaphoreCount = 1; present.pWaitSemaphores = &signal.semaphore;
        present.swapchainCount = 1; present.pSwapchains = &swapchain_handle; present.pImageIndices = &image_index;
        const VkFence present_fence = swapchain.prepare_present(image_index);
        VkSwapchainPresentFenceInfoEXT fence_info{VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_FENCE_INFO_EXT};
        fence_info.swapchainCount = 1;
        fence_info.pFences = &present_fence;
        if (present_fence) present.pNext = &fence_info;
        const auto result = vkQueuePresentKHR(device.present_queue(), &present);
        swapchain.presented(image_index, result);
        const bool present_needs_resize = result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR;
        if (!present_needs_resize) check_vk(result, "Present frame");
        resize_pending = present_needs_resize || acquire == VK_SUBOPTIMAL_KHR;
        current_frame = (current_frame + 1) % frame_count;
        return true;
    }
};
Viewer::Viewer(Platform& platform, ViewState& state, ViewerOptions options)
    : impl_(std::make_unique<Impl>(platform, state, options)) {}
Viewer::~Viewer() = default;
bool Viewer::draw(const InputState& input, float delta) { return impl_->draw(input, delta); }
void Viewer::request_resize() { impl_->resize_pending = true; }
void Viewer::wait_idle() { impl_->device.wait_idle(); }
unsigned Viewer::validation_errors() const { return impl_->device.validation_errors(); }
bool Viewer::wants_text_input() const { return impl_ && impl_->scene->wants_text_input(); }
unsigned Viewer::shutdown() {
    if (!impl_) return 0;
    auto errors = impl_->device.validation_counter();
    impl_.reset();
    return errors->load();
}
}
