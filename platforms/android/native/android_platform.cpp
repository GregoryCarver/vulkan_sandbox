#include "android_platform.hpp"
#include <sandbox/core/device.hpp>
#include <android/asset_manager.h>
#include <android/configuration.h>
#include <android/native_window.h>
#include <algorithm>
#include <limits>
#include <memory>
#include <stdexcept>

namespace sandbox::android {
std::vector<const char*> AndroidPlatform::instance_extensions() const {
    return {VK_KHR_SURFACE_EXTENSION_NAME, VK_KHR_ANDROID_SURFACE_EXTENSION_NAME};
}

VkSurfaceKHR AndroidPlatform::create_surface(VkInstance instance) const {
    if (!app_.window) throw std::runtime_error("Android window is not available");
    VkAndroidSurfaceCreateInfoKHR info{VK_STRUCTURE_TYPE_ANDROID_SURFACE_CREATE_INFO_KHR};
    info.window = app_.window;
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    check_vk(vkCreateAndroidSurfaceKHR(instance, &info, nullptr, &surface), "Create Android Vulkan surface");
    return surface;
}

VkExtent2D AndroidPlatform::framebuffer_extent() const {
    if (!app_.window) return {};
    return {static_cast<uint32_t>(std::max(0, ANativeWindow_getWidth(app_.window))),
            static_cast<uint32_t>(std::max(0, ANativeWindow_getHeight(app_.window)))};
}

float AndroidPlatform::content_scale() const {
    const int density = AConfiguration_getDensity(app_.config);
    if (density <= 0 || density == ACONFIGURATION_DENSITY_ANY || density == ACONFIGURATION_DENSITY_NONE)
        return 1.0f;
    return static_cast<float>(density) / 160.0f;
}

std::vector<std::byte> AndroidPlatform::read_asset(std::string_view path) const {
    const std::string name(path);
    if (name.empty() || name.front() == '/' || name.find("..") != std::string::npos)
        throw std::runtime_error("Asset name must be relative to the APK assets directory");
    using Asset = std::unique_ptr<AAsset, decltype(&AAsset_close)>;
    Asset asset(AAssetManager_open(app_.activity->assetManager, name.c_str(), AASSET_MODE_BUFFER), AAsset_close);
    if (!asset) throw std::runtime_error("Missing APK asset: " + name);
    const auto length = AAsset_getLength64(asset.get());
    if (length < 0 || static_cast<uint64_t>(length) > std::numeric_limits<size_t>::max())
        throw std::runtime_error("APK asset is too large: " + name);
    std::vector<std::byte> bytes(static_cast<size_t>(length));
    size_t offset = 0;
    while (offset != bytes.size()) {
        const size_t remaining = std::min(bytes.size() - offset, static_cast<size_t>(std::numeric_limits<int>::max()));
        const int count = AAsset_read(asset.get(), bytes.data() + offset, remaining);
        if (count <= 0) throw std::runtime_error("Could not read complete APK asset: " + name);
        offset += static_cast<size_t>(count);
    }
    return bytes;
}

std::string AndroidPlatform::writable_path(std::string_view filename) const {
    if (filename.empty() || filename.find('/') != std::string_view::npos || filename.find('\\') != std::string_view::npos)
        throw std::runtime_error("Expected an application settings filename");
    if (!app_.activity->internalDataPath || !*app_.activity->internalDataPath) return {};
    return std::string(app_.activity->internalDataPath) + "/" + std::string(filename);
}
}
