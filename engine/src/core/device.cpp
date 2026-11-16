#include <sandbox/core/device.hpp>
#include <sandbox/core/selection.hpp>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <vector>
#ifdef __ANDROID__
#include <android/log.h>
#if __ANDROID_MIN_SDK_VERSION__ < 33
#error "Direct Vulkan 1.3 linkage requires Android API 33 or newer."
#endif
#endif

namespace sandbox {
void check_vk(VkResult result, const char* operation) {
    if (result != VK_SUCCESS)
        throw std::runtime_error(std::string(operation) + " failed (VkResult " + std::to_string(result) + ")");
}
namespace {
void report(const std::string& message, bool warning = false) {
#ifdef __ANDROID__
    __android_log_write(warning ? ANDROID_LOG_WARN : ANDROID_LOG_INFO, "VulkanSandbox", message.c_str());
#else
    (warning ? std::cerr : std::cout) << message << '\n';
#endif
}
bool contains(const std::vector<VkExtensionProperties>& extensions, const char* name) {
    return std::ranges::any_of(extensions, [name](const auto& item) { return !std::strcmp(item.extensionName, name); });
}
}
VKAPI_ATTR VkBool32 VKAPI_CALL Device::debug_callback(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT, const VkDebugUtilsMessengerCallbackDataEXT* message, void* user) {
    if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)
        static_cast<Device*>(user)->validation_errors_->fetch_add(1);
    const char* text = message->pMessage ? message->pMessage : "Vulkan validation message";
#ifdef __ANDROID__
    __android_log_print(ANDROID_LOG_WARN, "VulkanSandbox", "%s", text);
#else
    std::cerr << "[Vulkan] " << text << '\n';
#endif
    return VK_FALSE;
}
Device::Device(Platform& platform, bool validation, bool present_fences) {
    try {
        uint32_t loader_version = VK_API_VERSION_1_0;
        check_vk(vkEnumerateInstanceVersion(&loader_version), "Query Vulkan loader version");
        if (loader_version < VK_API_VERSION_1_3)
            throw std::runtime_error("This template requires a Vulkan 1.3 loader and device");
        uint32_t count = 0;
        check_vk(vkEnumerateInstanceExtensionProperties(nullptr, &count, nullptr), "Count instance extensions");
        std::vector<VkExtensionProperties> extensions(count);
        check_vk(vkEnumerateInstanceExtensionProperties(nullptr, &count, extensions.data()), "Enumerate instance extensions");
        auto required = platform.instance_extensions();
        check_vk(vkEnumerateInstanceLayerProperties(&count, nullptr), "Count instance layers");
        std::vector<VkLayerProperties> layers(count);
        check_vk(vkEnumerateInstanceLayerProperties(&count, layers.data()), "Enumerate instance layers");
        const char* validation_layer = "VK_LAYER_KHRONOS_validation";
        validation_enabled_ = validation && std::ranges::any_of(layers, [validation_layer](const auto& layer) {
            return !std::strcmp(layer.layerName, validation_layer);
        });
        if (validation && !validation_enabled_)
            report("Validation requested, but VK_LAYER_KHRONOS_validation is unavailable", true);
        const bool debug_utils = validation_enabled_ && contains(extensions, VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        if (debug_utils) required.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        // Optional present fences make presentation-resource retirement explicit.
        // Both instance dependencies must be advertised before enabling this path.
        const bool surface_maintenance =
            contains(extensions, VK_EXT_SURFACE_MAINTENANCE_1_EXTENSION_NAME) &&
            contains(extensions, VK_KHR_GET_SURFACE_CAPABILITIES_2_EXTENSION_NAME);
        if (surface_maintenance) {
            required.push_back(VK_KHR_GET_SURFACE_CAPABILITIES_2_EXTENSION_NAME);
            required.push_back(VK_EXT_SURFACE_MAINTENANCE_1_EXTENSION_NAME);
        }
        VkInstanceCreateFlags flags = 0;
        if (contains(extensions, VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME)) {
            required.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
            flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
        }
        std::sort(required.begin(), required.end(), [](auto a, auto b) { return std::strcmp(a, b) < 0; });
        required.erase(std::unique(required.begin(), required.end(), [](auto a, auto b) { return !std::strcmp(a, b); }), required.end());
        for (const char* name : required)
            if (!contains(extensions, name)) throw std::runtime_error(std::string("Missing instance extension: ") + name);
        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        app.pApplicationName = "Vulkan Sandbox";
        app.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        app.pEngineName = "Sandbox";
        app.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        app.apiVersion = VK_API_VERSION_1_3;
        VkDebugUtilsMessengerCreateInfoEXT debug{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
        debug.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        debug.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        debug.pfnUserCallback = debug_callback;
        debug.pUserData = this;
        VkInstanceCreateInfo info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        info.flags = flags;
        info.pApplicationInfo = &app;
        info.enabledExtensionCount = static_cast<uint32_t>(required.size());
        info.ppEnabledExtensionNames = required.data();
        info.enabledLayerCount = validation_enabled_ ? 1u : 0u;
        info.ppEnabledLayerNames = validation_enabled_ ? &validation_layer : nullptr;
        info.pNext = debug_utils ? &debug : nullptr;
        check_vk(vkCreateInstance(&info, nullptr, &instance_), "Create Vulkan instance");
        if (debug_utils) {
            auto create = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(vkGetInstanceProcAddr(instance_, "vkCreateDebugUtilsMessengerEXT"));
            if (create) check_vk(create(instance_, &debug, nullptr, &messenger_), "Create validation messenger");
        }
        surface_ = platform.create_surface(instance_);
        if (!surface_) throw std::runtime_error("Platform returned a null Vulkan surface");
        check_vk(vkEnumeratePhysicalDevices(instance_, &count, nullptr), "Count GPUs");
        std::vector<VkPhysicalDevice> physical_devices(count);
        check_vk(vkEnumeratePhysicalDevices(instance_, &count, physical_devices.data()), "Enumerate GPUs");
        int best_score = -1;
        std::vector<const char*> selected_extensions;
        for (auto candidate : physical_devices) {
            VkPhysicalDeviceProperties properties{};
            vkGetPhysicalDeviceProperties(candidate, &properties);
            auto reject = [&](const char* why) { report(std::string("GPU ") + properties.deviceName + ": " + why, true); };
            if (properties.apiVersion < VK_API_VERSION_1_3) { reject("Vulkan 1.3 required"); continue; }
            VkPhysicalDeviceVulkan13Features features13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
            VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
            features.pNext = &features13;
            vkGetPhysicalDeviceFeatures2(candidate, &features);
            if (!features13.dynamicRendering || !features13.synchronization2) { reject("dynamic rendering and synchronization2 required"); continue; }
            check_vk(vkEnumerateDeviceExtensionProperties(candidate, nullptr, &count, nullptr), "Count device extensions");
            std::vector<VkExtensionProperties> available(count);
            check_vk(vkEnumerateDeviceExtensionProperties(candidate, nullptr, &count, available.data()), "Enumerate device extensions");
            if (!contains(available, VK_KHR_SWAPCHAIN_EXTENSION_NAME)) { reject("swapchain extension missing"); continue; }
            bool supports_present_fences = false;
            if (present_fences && surface_maintenance && contains(available, VK_EXT_SWAPCHAIN_MAINTENANCE_1_EXTENSION_NAME)) {
                VkPhysicalDeviceSwapchainMaintenance1FeaturesEXT maintenance{
                    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SWAPCHAIN_MAINTENANCE_1_FEATURES_EXT};
                VkPhysicalDeviceFeatures2 optional_features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
                optional_features.pNext = &maintenance;
                vkGetPhysicalDeviceFeatures2(candidate, &optional_features);
                supports_present_fences = maintenance.swapchainMaintenance1 == VK_TRUE;
            }
            vkGetPhysicalDeviceQueueFamilyProperties(candidate, &count, nullptr);
            std::vector<VkQueueFamilyProperties> families(count);
            vkGetPhysicalDeviceQueueFamilyProperties(candidate, &count, families.data());
            std::vector<QueueSupport> support;
            for (uint32_t i = 0; i < count; ++i) {
                VkBool32 present = VK_FALSE;
                check_vk(vkGetPhysicalDeviceSurfaceSupportKHR(candidate, i, surface_, &present), "Query presentation support");
                support.push_back({families[i].queueFlags, present != VK_FALSE, families[i].queueCount});
            }
            auto queues = select_queues(support);
            if (!queues) { reject("no graphics/presentation queues"); continue; }
            uint32_t formats = 0, modes = 0;
            check_vk(vkGetPhysicalDeviceSurfaceFormatsKHR(candidate, surface_, &formats, nullptr), "Count surface formats");
            check_vk(vkGetPhysicalDeviceSurfacePresentModesKHR(candidate, surface_, &modes, nullptr), "Count present modes");
            if (!formats || !modes) { reject("surface cannot present"); continue; }
            int score = properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? 100 : 10;
            if (queues->graphics == queues->present) score += 5;
            if (score <= best_score) continue;
            best_score = score;
            physical_ = candidate;
            properties_ = properties;
            graphics_family_ = queues->graphics;
            present_family_ = queues->present;
            selected_extensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
            present_fences_enabled_ = supports_present_fences;
            if (present_fences_enabled_) selected_extensions.push_back(VK_EXT_SWAPCHAIN_MAINTENANCE_1_EXTENSION_NAME);
            // Portability drivers require their subset extension to be enabled when advertised.
            if (contains(available, "VK_KHR_portability_subset")) selected_extensions.push_back("VK_KHR_portability_subset");
        }
        if (!physical_) throw std::runtime_error("No suitable GPU: Vulkan 1.3, dynamic rendering, synchronization2 and presentation are required");
        float priority = 1.0f;
        std::vector<VkDeviceQueueCreateInfo> queues;
        for (uint32_t family : std::set<uint32_t>{graphics_family_, present_family_}) {
            VkDeviceQueueCreateInfo queue{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
            queue.queueFamilyIndex = family;
            queue.queueCount = 1;
            queue.pQueuePriorities = &priority;
            queues.push_back(queue);
        }
        VkPhysicalDeviceVulkan13Features features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
        features.synchronization2 = VK_TRUE;
        features.dynamicRendering = VK_TRUE;
        VkPhysicalDeviceSwapchainMaintenance1FeaturesEXT maintenance{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SWAPCHAIN_MAINTENANCE_1_FEATURES_EXT};
        maintenance.swapchainMaintenance1 = VK_TRUE;
        if (present_fences_enabled_) features.pNext = &maintenance;
        VkDeviceCreateInfo create{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        create.pNext = &features;
        create.queueCreateInfoCount = static_cast<uint32_t>(queues.size());
        create.pQueueCreateInfos = queues.data();
        create.enabledExtensionCount = static_cast<uint32_t>(selected_extensions.size());
        create.ppEnabledExtensionNames = selected_extensions.data();
        check_vk(vkCreateDevice(physical_, &create, nullptr, &device_), "Create logical device");
        vkGetDeviceQueue(device_, graphics_family_, 0, &graphics_queue_);
        vkGetDeviceQueue(device_, present_family_, 0, &present_queue_);
        std::ostringstream summary;
        summary << "GPU: " << properties_.deviceName << " | Vulkan " << VK_VERSION_MAJOR(properties_.apiVersion)
                  << '.' << VK_VERSION_MINOR(properties_.apiVersion) << '.' << VK_VERSION_PATCH(properties_.apiVersion)
                  << " | validation " << (validation_enabled_ ? "on" : "off")
                  << " | present fences " << (present_fences_enabled_ ? "on" : "off");
        report(summary.str());
    } catch (...) { destroy(); throw; }
}
Device::~Device() { destroy(); }
void Device::wait_idle() const { if (device_) check_vk(vkDeviceWaitIdle(device_), "Wait for device"); }
void Device::destroy() noexcept {
    if (device_) { vkDeviceWaitIdle(device_); vkDestroyDevice(device_, nullptr); device_ = VK_NULL_HANDLE; }
    if (surface_) { vkDestroySurfaceKHR(instance_, surface_, nullptr); surface_ = VK_NULL_HANDLE; }
    if (messenger_) {
        auto destroy = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(vkGetInstanceProcAddr(instance_, "vkDestroyDebugUtilsMessengerEXT"));
        if (destroy) destroy(instance_, messenger_, nullptr);
        messenger_ = VK_NULL_HANDLE;
    }
    if (instance_) { vkDestroyInstance(instance_, nullptr); instance_ = VK_NULL_HANDLE; }
}
}
