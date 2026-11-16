#pragma once

#include <vulkan/vulkan.h>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

struct VmaAllocator_T;
struct VmaAllocation_T;

namespace sandbox {
class Device;

// Allocator must outlive every Buffer/Image/UploadQueue created with it.
class Allocator {
public:
    explicit Allocator(Device& device);
    ~Allocator();
    Allocator(const Allocator&) = delete;
    Allocator& operator=(const Allocator&) = delete;
    Device& device() const { return *device_; }
private:
    friend class Buffer;
    friend class Image;
    Device* device_;
    VmaAllocator_T* allocator_ = nullptr;
};

enum class MemoryUsage { Device, Upload, Readback };

// Move-only owners do not wait for GPU use. Retire them after the last using
// submission's fence, or keep them until the device is idle during shutdown.
class Buffer {
public:
    Buffer() = default;
    Buffer(Allocator& allocator, VkDeviceSize size, VkBufferUsageFlags usage,
        MemoryUsage memory = MemoryUsage::Device);
    ~Buffer();
    Buffer(Buffer&& other) noexcept;
    Buffer& operator=(Buffer&& other) noexcept;
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    VkBuffer handle() const { return buffer_; }
    VkDeviceSize size() const { return size_; }
    VkBufferUsageFlags usage() const { return usage_; }
    void* mapped() const { return mapped_; }
    explicit operator bool() const { return buffer_ != VK_NULL_HANDLE; }
    void write(std::span<const std::byte> bytes, VkDeviceSize offset = 0);
    void flush(VkDeviceSize offset = 0, VkDeviceSize size = VK_WHOLE_SIZE);
    void invalidate(VkDeviceSize offset = 0, VkDeviceSize size = VK_WHOLE_SIZE);

private:
    void reset() noexcept;
    VmaAllocator_T* allocator_ = nullptr;
    VmaAllocation_T* allocation_ = nullptr;
    VkBuffer buffer_ = VK_NULL_HANDLE;
    VkDeviceSize size_ = 0;
    VkBufferUsageFlags usage_ = 0;
    void* mapped_ = nullptr;
};

struct ImageDesc {
    VkExtent2D extent{};
    VkFormat format = VK_FORMAT_UNDEFINED;
    VkImageUsageFlags usage = 0;
    VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT;
    uint32_t mip_levels = 1;
    uint32_t layers = 1;
};

class Image {
public:
    Image() = default;
    Image(Allocator& allocator, ImageDesc description);
    ~Image();
    Image(Image&& other) noexcept;
    Image& operator=(Image&& other) noexcept;
    Image(const Image&) = delete;
    Image& operator=(const Image&) = delete;

    VkImage handle() const { return image_; }
    VkImageView view() const { return view_; }
    VkFormat format() const { return description_.format; }
    VkExtent2D extent() const { return description_.extent; }
    const ImageDesc& description() const { return description_; }
    explicit operator bool() const { return image_ != VK_NULL_HANDLE; }

private:
    void reset() noexcept;
    VmaAllocator_T* allocator_ = nullptr;
    VmaAllocation_T* allocation_ = nullptr;
    VkDevice device_ = VK_NULL_HANDLE;
    VkImage image_ = VK_NULL_HANDLE;
    VkImageView view_ = VK_NULL_HANDLE;
    ImageDesc description_{};
};

class Sampler {
public:
    Sampler() = default;
    explicit Sampler(Device& device, VkFilter filter = VK_FILTER_LINEAR,
        VkSamplerAddressMode address_mode = VK_SAMPLER_ADDRESS_MODE_REPEAT);
    ~Sampler();
    Sampler(Sampler&& other) noexcept;
    Sampler& operator=(Sampler&& other) noexcept;
    Sampler(const Sampler&) = delete;
    Sampler& operator=(const Sampler&) = delete;
    VkSampler handle() const { return sampler_; }
private:
    void reset() noexcept;
    VkDevice device_ = VK_NULL_HANDLE;
    VkSampler sampler_ = VK_NULL_HANDLE;
};

class UploadQueue;

// Record a batch, then submit it once. An unsubmitted batch releases its command
// buffer and staging allocations. The queue must outlive its active batches.
class UploadBatch {
public:
    ~UploadBatch();
    UploadBatch(UploadBatch&& other) noexcept;
    UploadBatch& operator=(UploadBatch&& other) noexcept;
    UploadBatch(const UploadBatch&) = delete;
    UploadBatch& operator=(const UploadBatch&) = delete;

    // Destination must include TRANSFER_DST usage and remain alive until the
    // ticket completes. Replacing an in-use resource requires a new allocation.
    void upload(Buffer& destination, std::span<const std::byte> bytes,
        VkDeviceSize offset = 0,
        VkPipelineStageFlags2 destination_stage = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        VkAccessFlags2 destination_access = VK_ACCESS_2_MEMORY_READ_BIT);
    // Initial upload only: one layer, one mip, RGBA8 UNORM or SRGB. Final layout
    // is SHADER_READ_ONLY_OPTIMAL; transitions synchronize graphics sampling.
    void upload_rgba8(Image& destination, std::span<const std::byte> bytes);

private:
    friend class UploadQueue;
    explicit UploadBatch(UploadQueue& owner, VkCommandBuffer command);
    void reset() noexcept;
    UploadQueue* owner_ = nullptr;
    VkCommandBuffer command_ = VK_NULL_HANDLE;
    std::vector<Buffer> staging_;
};

// Main/render-thread use only. The application externally serializes access to
// Device::graphics_queue() with rendering submissions. Uploads use that same
// queue, so the barriers also order subsequent drawing of the uploaded data.
class UploadQueue {
public:
    using Ticket = uint64_t;
    UploadQueue(Device& device, Allocator& allocator);
    ~UploadQueue();
    UploadQueue(const UploadQueue&) = delete;
    UploadQueue& operator=(const UploadQueue&) = delete;

    UploadBatch begin();
    Ticket submit(UploadBatch&& batch);
    void collect();
    bool is_complete(Ticket ticket);
    void wait(Ticket ticket);
    size_t pending_count() const;

private:
    friend class UploadBatch;
    void discard(VkCommandBuffer command) noexcept;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
