#pragma once
#include <array>
#include <string>
#include <vector>

namespace sandbox {
enum class Key { Tab, Left, Right, Up, Down, PageUp, PageDown, Home, End, Insert, Delete,
    Backspace, Space, Enter, Escape, A, C, V, X, Y, Z, Ctrl, Shift, Alt, Count };
struct InputEvent {
    enum class Type { Key, PointerButton };
    Type type = Type::Key;
    Key key = Key::Tab;
    unsigned button = 0;
    bool down = false;
};
struct InputState {
    // Preserve transitions (including press+release between two rendered frames).
    std::vector<InputEvent> events;
    std::array<bool, static_cast<unsigned>(Key::Count)> keys{};
    float pointer_x = 0, pointer_y = 0;
    bool pointer_down = false;
    bool pointer_right = false;
    bool focused = true;
    bool look_active = false;
    float look_dx = 0, look_dy = 0;
    float scroll_y = 0;
    float move_x = 0, move_y = 0, move_z = 0;
    bool toggle_ui = false;
    std::string text;
};
struct ViewState {
    std::array<float, 3> position{0.0f, 1.0f, 4.0f};
    float yaw = -90.0f, pitch = -12.0f;
    bool show_ui = true;
    bool show_demo = true;
    bool animate = true;
    bool stream_mesh_updates = false;
};
}
