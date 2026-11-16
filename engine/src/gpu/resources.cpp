#include <sandbox/gpu/resources.hpp>
#include <sandbox/core/device.hpp>
#include "vma_config.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <list>
#include <stdexcept>
#include <utility>

namespace sandbox {
namespace {
void check_range(VkDeviceSize allocation_size, VkDeviceSize offset, VkDeviceSize size) {
    if (offset > allocation_size || (size != VK_WHOLE_SIZE && size > allocation_size - offset))
        throw std::out_of_range("Buffer range exceeds allocation");
}
}

Allocator::Allocator(Device& device) : device_(&device) {
    VmaVulkanFunctions functions{};
    functions.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
    functions.vkGetDeviceProcAddr = vkGetDeviceProcAddr;
    VmaAllocatorCreateInfo info{};
    info.physicalDevice = device.physical();
    info.device = device.logical();
    info.instance = device.instance();
    info.vulkanApiVersion = VK_API_VERSION_1_3;
    info.pVulkanFunctions = &functions;
    check_vk(vmaCreateAllocator(&info, &allocator_), "Create Vulkan memory allocator");
}

Allocator::~Allocator() {
    if (allocator_) vmaDestroyAllocator(allocator_);
}

Buffer::Buffer(Allocator& allocator, VkDeviceSize size, VkBufferUsageFlags usage,
    MemoryUsage memory) : allocator_(allocator.allocator_), size_(size), usage_(usage) {
    if (size == 0) throw std::invalid_argument("Buffer size must be positive");
    VkBufferCreateInfo buffer_info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    buffer_info.size = size;
    buffer_info.usage = usage;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VmaAllocationCreateInfo allocation_info{};
    allocation_info.usage = memory == MemoryUsage::Device ? VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE : VMA_MEMORY_USAGE_AUTO;
    if (memory != MemoryUsage::Device) {
        allocation_info.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT |
            (memory == MemoryUsage::Upload ? VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
                                          : VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT);
    }
    VmaAllocationInfo allocation{};
    check_vk(vmaCreateBuffer(allocator_, &buffer_info, &allocation_info, &buffer_, &allocation_, &allocation),
        "Allocate buffer");
    mapped_ = allocation.pMappedData;
}

Buffer::~Buffer() { reset(); }
Buffer::Buffer(Buffer&& other) noexcept { *this = std::move(other); }
Buffer& Buffer::operator=(Buffer&& other) noexcept {
    if (this != &other) {
        reset();
        allocator_ = std::exchange(other.allocator_, nullptr);
        allocation_ = std::exchange(other.allocation_, nullptr);
        buffer_ = std::exchange(other.buffer_, VK_NULL_HANDLE);
        size_ = std::exchange(other.size_, 0);
        usage_ = std::exchange(other.usage_, 0);
        mapped_ = std::exchange(other.mapped_, nullptr);
    }
    return *this;
}
void Buffer::reset() noexcept {
    if (buffer_) vmaDestroyBuffer(allocator_, buffer_, allocation_);
    allocator_ = nullptr;
    allocation_ = nullptr;
    buffer_ = VK_NULL_HANDLE;
    size_ = 0;
    usage_ = 0;
    mapped_ = nullptr;
}
void Buffer::write(std::span<const std::byte> bytes, VkDeviceSize offset) {
    check_range(size_, offset, bytes.size());
    if (!mapped_) throw std::logic_error("Buffer is not host-mapped");
    if (bytes.empty()) return;
    std::memcpy(static_cast<std::byte*>(mapped_) + offset, bytes.data(), bytes.size());
    flush(offset, bytes.size());
}
void Buffer::flush(VkDeviceSize offset, VkDeviceSize size) {
    check_range(size_, offset, size);
    if (!mapped_) throw std::logic_error("Buffer is not host-mapped");
    check_vk(vmaFlushAllocation(allocator_, allocation_, offset, size), "Flush buffer memory");
}
void Buffer::invalidate(VkDeviceSize offset, VkDeviceSize size) {
    check_range(size_, offset, size);
    if (!mapped_) throw std::logic_error("Buffer is not host-mapped");
    check_vk(vmaInvalidateAllocation(allocator_, allocation_, offset, size), "Invalidate buffer memory");
}

Image::Image(Allocator& allocator, ImageDesc description)
    : allocator_(allocator.allocator_), device_(allocator.device().logical()), description_(description) {
    if (description.extent.width == 0 || description.extent.height == 0 ||
        description.mip_levels == 0 || description.layers == 0)
        throw std::invalid_argument("Image dimensions, mip levels, and layers must be positive");
    VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = description.format;
    info.extent = {description.extent.width, description.extent.height, 1};
    info.mipLevels = description.mip_levels;
    info.arrayLayers = description.layers;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = description.usage;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VmaAllocationCreateInfo allocation_info{};
    allocation_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    check_vk(vmaCreateImage(allocator_, &info, &allocation_info, &image_, &allocation_, nullptr), "Allocate image");
    try {
        VkImageViewCreateInfo view_info{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        view_info.image = image_;
        view_info.viewType = description.layers == 1 ? VK_IMAGE_VIEW_TYPE_2D : VK_IMAGE_VIEW_TYPE_2D_ARRAY;
        view_info.format = description.format;
        view_info.subresourceRange = {description.aspect, 0, description.mip_levels, 0, description.layers};
        check_vk(vkCreateImageView(device_, &view_info, nullptr, &view_), "Create image view");
    } catch (...) {
        reset();
        throw;
    }
}
Image::~Image() { reset(); }
Image::Image(Image&& other) noexcept { *this = std::move(other); }
Image& Image::operator=(Image&& other) noexcept {
    if (this != &other) {
        reset();
        allocator_ = std::exchange(other.allocator_, nullptr);
        allocation_ = std::exchange(other.allocation_, nullptr);
        device_ = std::exchange(other.device_, VK_NULL_HANDLE);
        image_ = std::exchange(other.image_, VK_NULL_HANDLE);
        view_ = std::exchange(other.view_, VK_NULL_HANDLE);
        description_ = std::exchange(other.description_, {});
    }
    return *this;
}
void Image::reset() noexcept {
    if (view_) vkDestroyImageView(device_, view_, nullptr);
    if (image_) vmaDestroyImage(allocator_, image_, allocation_);
    allocator_ = nullptr;
    allocation_ = nullptr;
    device_ = VK_NULL_HANDLE;
    image_ = VK_NULL_HANDLE;
    view_ = VK_NULL_HANDLE;
    description_ = {};
}

Sampler::Sampler(Device& device, VkFilter filter, VkSamplerAddressMode address_mode)
    : device_(device.logical()) {
    VkSamplerCreateInfo info{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    info.magFilter = filter;
    info.minFilter = filter;
    info.mipmapMode = filter == VK_FILTER_LINEAR ? VK_SAMPLER_MIPMAP_MODE_LINEAR : VK_SAMPLER_MIPMAP_MODE_NEAREST;
    info.addressModeU = address_mode;
    info.addressModeV = address_mode;
    info.addressModeW = address_mode;
    info.minLod = 0.0f;
    info.maxLod = VK_LOD_CLAMP_NONE;
    info.maxAnisotropy = 1.0f;
    check_vk(vkCreateSampler(device_, &info, nullptr, &sampler_), "Create sampler");
}
Sampler::~Sampler() { reset(); }
Sampler::Sampler(Sampler&& other) noexcept { *this = std::move(other); }
Sampler& Sampler::operator=(Sampler&& other) noexcept {
    if (this != &other) {
        reset();
        device_ = std::exchange(other.device_, VK_NULL_HANDLE);
        sampler_ = std::exchange(other.sampler_, VK_NULL_HANDLE);
    }
    return *this;
}
void Sampler::reset() noexcept {
    if (sampler_) vkDestroySampler(device_, sampler_, nullptr);
    device_ = VK_NULL_HANDLE;
    sampler_ = VK_NULL_HANDLE;
}

struct UploadQueue::Impl {
    struct Submission {
        Ticket ticket = 0;
        VkFence fence = VK_NULL_HANDLE;
        VkCommandBuffer command = VK_NULL_HANDLE;
        std::vector<Buffer> staging;
    };
    Device* device;
    Allocator* allocator;
    VkCommandPool pool = VK_NULL_HANDLE;
    Ticket next_ticket = 1;
    Ticket completed_ticket = 0;
    std::list<Submission> pending;

    void release(Submission& submission) noexcept {
        if (submission.fence) vkDestroyFence(device->logical(), submission.fence, nullptr);
        if (submission.command) vkFreeCommandBuffers(device->logical(), pool, 1, &submission.command);
        submission.staging.clear();
    }
};

UploadQueue::UploadQueue(Device& device, Allocator& allocator) : impl_(std::make_unique<Impl>()) {
    if (&allocator.device() != &device) throw std::invalid_argument("Upload allocator belongs to another device");
    impl_->device = &device;
    impl_->allocator = &allocator;
    VkCommandPoolCreateInfo info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    info.queueFamilyIndex = device.graphics_family();
    check_vk(vkCreateCommandPool(device.logical(), &info, nullptr, &impl_->pool), "Create upload command pool");
}
UploadQueue::~UploadQueue() {
    // Final cleanup waits only outstanding uploads. The host must separately
    // wait for rendering before destroying the uploaded destination resources.
    for (auto& submission : impl_->pending) {
        (void)vkWaitForFences(impl_->device->logical(), 1, &submission.fence, VK_TRUE, UINT64_MAX);
        impl_->release(submission);
    }
    if (impl_->pool) vkDestroyCommandPool(impl_->device->logical(), impl_->pool, nullptr);
}
UploadBatch UploadQueue::begin() {
    collect();
    VkCommandBufferAllocateInfo allocation{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocation.commandPool = impl_->pool;
    allocation.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocation.commandBufferCount = 1;
    VkCommandBuffer command = VK_NULL_HANDLE;
    check_vk(vkAllocateCommandBuffers(impl_->device->logical(), &allocation, &command), "Allocate upload command buffer");
    UploadBatch batch(*this, command);
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    check_vk(vkBeginCommandBuffer(command, &begin), "Begin upload batch");
    return batch;
}
UploadQueue::Ticket UploadQueue::submit(UploadBatch&& batch) {
    if (batch.owner_ != this || !batch.command_) throw std::invalid_argument("Upload batch belongs to another queue or was submitted");
    check_vk(vkEndCommandBuffer(batch.command_), "End upload batch");
    // Allocate bookkeeping before submitting: no potentially-throwing container
    // allocation may discard staging memory after the GPU starts using it.
    impl_->pending.emplace_back();
    auto& submission = impl_->pending.back();
    VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    const VkResult fence_result = vkCreateFence(impl_->device->logical(), &fence_info, nullptr, &submission.fence);
    if (fence_result != VK_SUCCESS) {
        impl_->pending.pop_back();
        check_vk(fence_result, "Create upload fence");
    }
    submission.command = std::exchange(batch.command_, VK_NULL_HANDLE);
    submission.staging = std::move(batch.staging_);
    batch.owner_ = nullptr;
    submission.ticket = impl_->next_ticket;
    VkCommandBufferSubmitInfo command_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
    command_info.commandBuffer = submission.command;
    VkSubmitInfo2 submit_info{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
    submit_info.commandBufferInfoCount = 1;
    submit_info.pCommandBufferInfos = &command_info;
    const VkResult result = vkQueueSubmit2(impl_->device->graphics_queue(), 1, &submit_info, submission.fence);
    if (result != VK_SUCCESS) {
        impl_->release(submission);
        impl_->pending.pop_back();
        check_vk(result, "Submit upload batch");
    }
    return impl_->next_ticket++;
}
void UploadQueue::collect() {
    while (!impl_->pending.empty()) {
        auto& submission = impl_->pending.front();
        const VkResult result = vkGetFenceStatus(impl_->device->logical(), submission.fence);
        if (result == VK_NOT_READY) break;
        check_vk(result, "Check upload fence");
        impl_->completed_ticket = submission.ticket;
        impl_->release(submission);
        impl_->pending.pop_front();
    }
}
bool UploadQueue::is_complete(Ticket ticket) {
    if (ticket >= impl_->next_ticket) throw std::invalid_argument("Unknown upload ticket");
    collect();
    return ticket <= impl_->completed_ticket;
}
void UploadQueue::wait(Ticket ticket) {
    if (ticket >= impl_->next_ticket) throw std::invalid_argument("Unknown upload ticket");
    if (ticket <= impl_->completed_ticket) return;
    const auto submission = std::find_if(impl_->pending.begin(), impl_->pending.end(),
        [ticket](const auto& item) { return item.ticket == ticket; });
    if (submission == impl_->pending.end()) throw std::logic_error("Upload ticket bookkeeping mismatch");
    check_vk(vkWaitForFences(impl_->device->logical(), 1, &submission->fence, VK_TRUE, UINT64_MAX), "Wait for upload batch");
    collect();
}
size_t UploadQueue::pending_count() const { return impl_->pending.size(); }
void UploadQueue::discard(VkCommandBuffer command) noexcept {
    if (command) vkFreeCommandBuffers(impl_->device->logical(), impl_->pool, 1, &command);
}

UploadBatch::UploadBatch(UploadQueue& owner, VkCommandBuffer command) : owner_(&owner), command_(command) {}
UploadBatch::~UploadBatch() { reset(); }
UploadBatch::UploadBatch(UploadBatch&& other) noexcept { *this = std::move(other); }
UploadBatch& UploadBatch::operator=(UploadBatch&& other) noexcept {
    if (this != &other) {
        reset();
        owner_ = std::exchange(other.owner_, nullptr);
        command_ = std::exchange(other.command_, VK_NULL_HANDLE);
        staging_ = std::move(other.staging_);
    }
    return *this;
}
void UploadBatch::reset() noexcept {
    if (owner_ && command_) owner_->discard(command_);
    owner_ = nullptr;
    command_ = VK_NULL_HANDLE;
    staging_.clear();
}
void UploadBatch::upload(Buffer& destination, std::span<const std::byte> bytes, VkDeviceSize offset,
    VkPipelineStageFlags2 destination_stage, VkAccessFlags2 destination_access) {
    if (!owner_ || !command_) throw std::logic_error("Upload batch is no longer recording");
    if (!destination || !(destination.usage() & VK_BUFFER_USAGE_TRANSFER_DST_BIT))
        throw std::invalid_argument("Upload destination buffer requires TRANSFER_DST usage");
    check_range(destination.size(), offset, bytes.size());
    if (bytes.empty()) return;
    if ((offset % 4) != 0 || (bytes.size() % 4) != 0)
        throw std::invalid_argument("Vulkan buffer copies require four-byte-aligned offsets and sizes");
    Buffer staging(*owner_->impl_->allocator, bytes.size(), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, MemoryUsage::Upload);
    staging.write(bytes);
    staging_.push_back(std::move(staging));
    const VkBufferCopy copy{0, offset, bytes.size()};
    vkCmdCopyBuffer(command_, staging_.back().handle(), destination.handle(), 1, &copy);
    VkBufferMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2};
    barrier.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
    barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    barrier.dstStageMask = destination_stage;
    barrier.dstAccessMask = destination_access;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.buffer = destination.handle();
    barrier.offset = offset;
    barrier.size = bytes.size();
    VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dependency.bufferMemoryBarrierCount = 1;
    dependency.pBufferMemoryBarriers = &barrier;
    vkCmdPipelineBarrier2(command_, &dependency);
}
void UploadBatch::upload_rgba8(Image& destination, std::span<const std::byte> bytes) {
    if (!owner_ || !command_) throw std::logic_error("Upload batch is no longer recording");
    const auto& description = destination.description();
    if (!destination || !(description.usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT) || !(description.usage & VK_IMAGE_USAGE_SAMPLED_BIT))
        throw std::invalid_argument("RGBA upload destination requires TRANSFER_DST and SAMPLED usage");
    if ((description.format != VK_FORMAT_R8G8B8A8_UNORM && description.format != VK_FORMAT_R8G8B8A8_SRGB) ||
        description.layers != 1 || description.mip_levels != 1 || description.aspect != VK_IMAGE_ASPECT_COLOR_BIT)
        throw std::invalid_argument("RGBA upload requires a single-layer, single-mip RGBA8 color image");
    const uint64_t expected_size = static_cast<uint64_t>(description.extent.width) * description.extent.height * 4;
    if (bytes.size() != expected_size) throw std::invalid_argument("RGBA image upload byte count does not match extent");
    Buffer staging(*owner_->impl_->allocator, bytes.size(), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, MemoryUsage::Upload);
    staging.write(bytes);
    staging_.push_back(std::move(staging));
    VkImageMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    barrier.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
    barrier.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = destination.handle();
    barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dependency.imageMemoryBarrierCount = 1;
    dependency.pImageMemoryBarriers = &barrier;
    vkCmdPipelineBarrier2(command_, &dependency);
    VkBufferImageCopy copy{};
    copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    copy.imageExtent = {description.extent.width, description.extent.height, 1};
    vkCmdCopyBufferToImage(command_, staging_.back().handle(), destination.handle(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
    barrier.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
    barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    barrier.dstStageMask = VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT;
    barrier.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    vkCmdPipelineBarrier2(command_, &dependency);
}
}
