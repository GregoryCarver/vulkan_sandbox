#pragma once

#include <sandbox/viewer/input.hpp>
#include <imgui.h>
#include <array>

namespace sandbox::detail {
inline constexpr std::array<ImGuiKey, static_cast<unsigned>(Key::Count)> imgui_keys{{
    ImGuiKey_Tab, ImGuiKey_LeftArrow, ImGuiKey_RightArrow, ImGuiKey_UpArrow,
    ImGuiKey_DownArrow, ImGuiKey_PageUp, ImGuiKey_PageDown, ImGuiKey_Home,
    ImGuiKey_End, ImGuiKey_Insert, ImGuiKey_Delete, ImGuiKey_Backspace,
    ImGuiKey_Space, ImGuiKey_Enter, ImGuiKey_Escape, ImGuiKey_A, ImGuiKey_C,
    ImGuiKey_V, ImGuiKey_X, ImGuiKey_Y, ImGuiKey_Z, ImGuiMod_Ctrl, ImGuiMod_Shift, ImGuiMod_Alt}};

inline void feed_imgui_input(ImGuiIO& io, const InputState& input) {
    io.AddFocusEvent(input.focused);
    io.AddMousePosEvent(input.pointer_x, input.pointer_y);
    // Preserve transitions before reconciling the final held-state snapshot.
    // ImGui trickles a quick press/release over frames so widgets see both.
    for (const auto& event : input.events) {
        if (event.type == InputEvent::Type::Key) {
            const auto key = static_cast<unsigned>(event.key);
            if (key < imgui_keys.size()) io.AddKeyEvent(imgui_keys[key], event.down);
        } else if (event.button < 5) {
            io.AddMouseButtonEvent(static_cast<int>(event.button), event.down);
        }
    }
    io.AddMouseButtonEvent(0, input.pointer_down);
    io.AddMouseButtonEvent(1, input.pointer_right);
    for (unsigned index = 0; index < static_cast<unsigned>(Key::Count); ++index)
        io.AddKeyEvent(imgui_keys[index], input.keys[index]);
    io.AddMouseWheelEvent(0, input.scroll_y);
    if (!input.text.empty()) io.AddInputCharactersUTF8(input.text.c_str());
}
}
