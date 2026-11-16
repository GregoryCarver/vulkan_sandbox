#include "windows_platform.hpp"
#include <sandbox/core/device.hpp>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <windows.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <utility>

namespace sandbox {
namespace {
constexpr std::array<int, static_cast<unsigned>(Key::Ctrl)> glfw_keys{
    GLFW_KEY_TAB, GLFW_KEY_LEFT, GLFW_KEY_RIGHT, GLFW_KEY_UP, GLFW_KEY_DOWN,
    GLFW_KEY_PAGE_UP, GLFW_KEY_PAGE_DOWN, GLFW_KEY_HOME, GLFW_KEY_END, GLFW_KEY_INSERT,
    GLFW_KEY_DELETE, GLFW_KEY_BACKSPACE, GLFW_KEY_SPACE, GLFW_KEY_ENTER, GLFW_KEY_ESCAPE,
    GLFW_KEY_A, GLFW_KEY_C, GLFW_KEY_V, GLFW_KEY_X, GLFW_KEY_Y, GLFW_KEY_Z};

void record_modifiers(InputState& input, int modifiers) {
    constexpr std::array<Key, 3> keys{Key::Ctrl, Key::Shift, Key::Alt};
    constexpr std::array<int, 3> flags{GLFW_MOD_CONTROL, GLFW_MOD_SHIFT, GLFW_MOD_ALT};
    for (size_t index = 0; index < keys.size(); ++index) {
        const bool down = (modifiers & flags[index]) != 0;
        auto& held = input.keys[static_cast<unsigned>(keys[index])];
        if (held != down) {
            input.events.push_back({InputEvent::Type::Key, keys[index], 0, down});
            held = down;
        }
    }
}

std::filesystem::path executable_directory() {
    std::wstring path(512, L'\0');
    for (;;) {
        auto size = GetModuleFileNameW(nullptr, path.data(), static_cast<DWORD>(path.size()));
        if (!size) throw std::runtime_error("Cannot determine executable path");
        if (size < path.size()) { path.resize(size); return std::filesystem::path(path).parent_path(); }
        path.resize(path.size() * 2);
    }
}
void append_utf8(std::string& text, unsigned codepoint) {
    if (codepoint <= 0x7f) text += static_cast<char>(codepoint);
    else if (codepoint <= 0x7ff) {
        text += static_cast<char>(0xc0 | (codepoint >> 6));
        text += static_cast<char>(0x80 | (codepoint & 0x3f));
    } else if (codepoint <= 0xffff) {
        text += static_cast<char>(0xe0 | (codepoint >> 12));
        text += static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f));
        text += static_cast<char>(0x80 | (codepoint & 0x3f));
    } else {
        text += static_cast<char>(0xf0 | (codepoint >> 18));
        text += static_cast<char>(0x80 | ((codepoint >> 12) & 0x3f));
        text += static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f));
        text += static_cast<char>(0x80 | (codepoint & 0x3f));
    }
}
}
WindowsPlatform::WindowsPlatform(int width, int height, bool hidden, std::filesystem::path asset_override) {
    glfwSetErrorCallback([](int code, const char* description) { std::cerr << "GLFW " << code << ": " << description << '\n'; });
    if (!glfwInit()) throw std::runtime_error("GLFW initialization failed");
    try {
        if (!glfwVulkanSupported()) throw std::runtime_error("Install a GPU driver with Vulkan support");
        assets_ = asset_override.empty() ? executable_directory() / "assets" : std::filesystem::absolute(asset_override);
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_VISIBLE, hidden ? GLFW_FALSE : GLFW_TRUE);
        window_ = glfwCreateWindow(width, height, "Vulkan Sandbox | Windows + Android", nullptr, nullptr);
        if (!window_) throw std::runtime_error("Window creation failed");
        glfwSetWindowUserPointer(window_, this);
        glfwSetFramebufferSizeCallback(window_, [](GLFWwindow* window, int, int) {
            static_cast<WindowsPlatform*>(glfwGetWindowUserPointer(window))->resized_ = true;
        });
        glfwSetScrollCallback(window_, [](GLFWwindow* window, double, double y) {
            static_cast<WindowsPlatform*>(glfwGetWindowUserPointer(window))->input_.scroll_y += static_cast<float>(y);
        });
        glfwSetCharCallback(window_, [](GLFWwindow* window, unsigned character) {
            append_utf8(static_cast<WindowsPlatform*>(glfwGetWindowUserPointer(window))->input_.text, character);
        });
        glfwSetKeyCallback(window_, [](GLFWwindow* window, int key, int, int action, int modifiers) {
            auto& input = static_cast<WindowsPlatform*>(glfwGetWindowUserPointer(window))->input_;
            record_modifiers(input, modifiers);
            if (key == GLFW_KEY_F1 && action == GLFW_PRESS)
                input.toggle_ui = true;
            if (action != GLFW_PRESS && action != GLFW_RELEASE) return;
            const auto found = std::find(glfw_keys.begin(), glfw_keys.end(), key);
            if (found != glfw_keys.end()) {
                const auto index = static_cast<unsigned>(found - glfw_keys.begin());
                const bool down = action == GLFW_PRESS;
                input.events.push_back({InputEvent::Type::Key, static_cast<Key>(index), 0, down});
                input.keys[index] = down;
            }
        });
        glfwSetMouseButtonCallback(window_, [](GLFWwindow* window, int button, int action, int modifiers) {
            auto& input = static_cast<WindowsPlatform*>(glfwGetWindowUserPointer(window))->input_;
            record_modifiers(input, modifiers);
            if (button >= 0 && button < 5 && (action == GLFW_PRESS || action == GLFW_RELEASE))
                input.events.push_back({InputEvent::Type::PointerButton, Key::Tab,
                                        static_cast<unsigned>(button), action == GLFW_PRESS});
        });
    } catch (...) {
        if (window_) glfwDestroyWindow(window_);
        glfwTerminate();
        throw;
    }
}
WindowsPlatform::~WindowsPlatform() { if (window_) glfwDestroyWindow(window_); glfwTerminate(); }
std::vector<const char*> WindowsPlatform::instance_extensions() const {
    uint32_t count = 0;
    const char** extensions = glfwGetRequiredInstanceExtensions(&count);
    if (!extensions || !count) throw std::runtime_error("GLFW returned no Vulkan surface extensions");
    return {extensions, extensions + count};
}
VkSurfaceKHR WindowsPlatform::create_surface(VkInstance instance) const {
    VkSurfaceKHR surface{};
    check_vk(glfwCreateWindowSurface(instance, window_, nullptr, &surface), "Create Windows surface");
    return surface;
}
VkExtent2D WindowsPlatform::framebuffer_extent() const {
    if (glfwGetWindowAttrib(window_, GLFW_ICONIFIED)) return {};
    int width = 0, height = 0;
    glfwGetFramebufferSize(window_, &width, &height);
    return {static_cast<uint32_t>(std::max(0, width)), static_cast<uint32_t>(std::max(0, height))};
}
float WindowsPlatform::content_scale() const {
    float x = 1, y = 1;
    glfwGetWindowContentScale(window_, &x, &y);
    return std::max(x, y);
}
std::vector<std::byte> WindowsPlatform::read_asset(std::string_view name) const {
    auto relative = std::filesystem::path(name);
    if (relative.is_absolute()) throw std::runtime_error("Asset paths must be relative");
    for (const auto& part : relative) if (part == "..") throw std::runtime_error("Asset path escapes asset directory");
    const auto path = assets_ / relative;
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("Cannot read asset: " + path.string());
    const auto length = file.tellg();
    if (length < 0) throw std::runtime_error("Cannot determine asset size: " + path.string());
    std::vector<std::byte> bytes(static_cast<size_t>(length));
    file.seekg(0);
    if (!file.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size())))
        throw std::runtime_error("Incomplete asset read: " + path.string());
    return bytes;
}
std::string WindowsPlatform::writable_path(std::string_view filename) const {
    // User settings are independent from installation and packaged assets.
    std::wstring base(32768, L'\0');
    auto count = GetEnvironmentVariableW(L"LOCALAPPDATA", base.data(), static_cast<DWORD>(base.size()));
    if (!count || count >= base.size()) return {};
    base.resize(count);
    auto directory = std::filesystem::path(base) / "VulkanSandbox";
    std::error_code error;
    std::filesystem::create_directories(directory, error);
    if (error) return {};
    const auto utf8 = (directory / std::filesystem::path(filename)).u8string();
    return std::string(reinterpret_cast<const char*>(utf8.data()), utf8.size());
}
InputState WindowsPlatform::poll_input() {
    glfwPollEvents();
    auto pressed = [&](int key) { return glfwGetKey(window_, key) == GLFW_PRESS; };
    for (size_t i = 0; i < glfw_keys.size(); ++i) input_.keys[i] = pressed(glfw_keys[i]);
    input_.keys[static_cast<unsigned>(Key::Ctrl)] = pressed(GLFW_KEY_LEFT_CONTROL) || pressed(GLFW_KEY_RIGHT_CONTROL);
    input_.keys[static_cast<unsigned>(Key::Shift)] = pressed(GLFW_KEY_LEFT_SHIFT) || pressed(GLFW_KEY_RIGHT_SHIFT);
    input_.keys[static_cast<unsigned>(Key::Alt)] = pressed(GLFW_KEY_LEFT_ALT) || pressed(GLFW_KEY_RIGHT_ALT);
    input_.focused = glfwGetWindowAttrib(window_, GLFW_FOCUSED) != 0;
    double x = 0, y = 0;
    glfwGetCursorPos(window_, &x, &y);
    int width = 1, height = 1, fb_width = 1, fb_height = 1;
    glfwGetWindowSize(window_, &width, &height);
    glfwGetFramebufferSize(window_, &fb_width, &fb_height);
    const float scale_x = static_cast<float>(fb_width) / std::max(width, 1);
    const float scale_y = static_cast<float>(fb_height) / std::max(height, 1);
    input_.pointer_x = static_cast<float>(x) * scale_x;
    input_.pointer_y = static_cast<float>(y) * scale_y;
    input_.pointer_down = glfwGetMouseButton(window_, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
    input_.pointer_right = glfwGetMouseButton(window_, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;
    input_.look_active = input_.focused && input_.pointer_right;
    input_.look_dx = input_.look_active && was_looking_ ? static_cast<float>(x - previous_x_) : 0;
    input_.look_dy = input_.look_active && was_looking_ ? static_cast<float>(y - previous_y_) : 0;
    previous_x_ = x; previous_y_ = y; was_looking_ = input_.look_active;
    input_.move_x = static_cast<float>(pressed(GLFW_KEY_D)) - static_cast<float>(pressed(GLFW_KEY_A));
    input_.move_y = static_cast<float>(pressed(GLFW_KEY_E)) - static_cast<float>(pressed(GLFW_KEY_Q));
    input_.move_z = static_cast<float>(pressed(GLFW_KEY_W)) - static_cast<float>(pressed(GLFW_KEY_S));
    auto result = input_;
    input_.events.clear(); input_.text.clear(); input_.scroll_y = 0; input_.toggle_ui = false;
    return result;
}
bool WindowsPlatform::take_resize() { return std::exchange(resized_, false); }
bool WindowsPlatform::should_close() const { return glfwWindowShouldClose(window_) != 0; }
}
