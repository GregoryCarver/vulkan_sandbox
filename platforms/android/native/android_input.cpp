#include "android_input.hpp"
#include <android/input.h>
#include <android/keycodes.h>
#include <algorithm>
#include <optional>

namespace sandbox::android {
namespace {
std::optional<Key> translate_key(int32_t code) {
    switch (code) {
    case AKEYCODE_TAB: return Key::Tab;
    case AKEYCODE_DPAD_LEFT: return Key::Left;
    case AKEYCODE_DPAD_RIGHT: return Key::Right;
    case AKEYCODE_DPAD_UP: return Key::Up;
    case AKEYCODE_DPAD_DOWN: return Key::Down;
    case AKEYCODE_PAGE_UP: return Key::PageUp;
    case AKEYCODE_PAGE_DOWN: return Key::PageDown;
    case AKEYCODE_MOVE_HOME: return Key::Home;
    case AKEYCODE_MOVE_END: return Key::End;
    case AKEYCODE_INSERT: return Key::Insert;
    case AKEYCODE_FORWARD_DEL: return Key::Delete;
    case AKEYCODE_DEL: return Key::Backspace;
    case AKEYCODE_SPACE: return Key::Space;
    case AKEYCODE_ENTER: case AKEYCODE_NUMPAD_ENTER: return Key::Enter;
    case AKEYCODE_ESCAPE: return Key::Escape;
    case AKEYCODE_A: return Key::A;
    case AKEYCODE_C: return Key::C;
    case AKEYCODE_V: return Key::V;
    case AKEYCODE_X: return Key::X;
    case AKEYCODE_Y: return Key::Y;
    case AKEYCODE_Z: return Key::Z;
    case AKEYCODE_CTRL_LEFT: case AKEYCODE_CTRL_RIGHT: return Key::Ctrl;
    case AKEYCODE_SHIFT_LEFT: case AKEYCODE_SHIFT_RIGHT: return Key::Shift;
    case AKEYCODE_ALT_LEFT: case AKEYCODE_ALT_RIGHT: return Key::Alt;
    default: return std::nullopt;
    }
}
}

void AndroidInput::set_pointer(bool down) {
    if (state.pointer_down != down) {
        state.events.push_back({InputEvent::Type::PointerButton, Key::Tab, 0, down});
        state.pointer_down = down;
    }
}

void AndroidInput::key(const GameActivityKeyEvent& event) {
    const bool down = event.action == AKEY_EVENT_ACTION_DOWN;
    if (event.action != AKEY_EVENT_ACTION_DOWN && event.action != AKEY_EVENT_ACTION_UP) return;
    if (const auto mapped = translate_key(event.keyCode)) {
        if (state.keys[static_cast<unsigned>(*mapped)] != down)
            state.events.push_back({InputEvent::Type::Key, *mapped, 0, down});
        state.keys[static_cast<unsigned>(*mapped)] = down;
    }
    switch (event.keyCode) {
    case AKEYCODE_W: movement_keys_[0] = down; break;
    case AKEYCODE_S: movement_keys_[1] = down; break;
    case AKEYCODE_A: movement_keys_[2] = down; break;
    case AKEYCODE_D: movement_keys_[3] = down; break;
    case AKEYCODE_Q: movement_keys_[4] = down; break;
    case AKEYCODE_E: movement_keys_[5] = down; break;
    case AKEYCODE_F1: if (down && event.repeatCount == 0) state.toggle_ui = true; break;
    default: break;
    }
}

void AndroidInput::motion(const GameActivityMotionEvent& event, float width, float height, float scale) {
    const int32_t action = event.action & AMOTION_EVENT_ACTION_MASK;
    if (action == AMOTION_EVENT_ACTION_CANCEL) { release_all(); return; }
    if (event.pointerCount == 0) return;

    const bool mouse = (event.source & AINPUT_SOURCE_MOUSE) == AINPUT_SOURCE_MOUSE;
    if (mouse) {
        const float x = GameActivityPointerAxes_getAxisValue(&event.pointers[0], AMOTION_EVENT_AXIS_X);
        const float y = GameActivityPointerAxes_getAxisValue(&event.pointers[0], AMOTION_EVENT_AXIS_Y);
        const bool right = (event.buttonState & AMOTION_EVENT_BUTTON_SECONDARY) != 0;
        if (right && state.pointer_right) {
            state.look_dx += x - state.pointer_x;
            state.look_dy += y - state.pointer_y;
        }
        state.pointer_x = x;
        state.pointer_y = y;
        set_pointer((event.buttonState & AMOTION_EVENT_BUTTON_PRIMARY) != 0);
        if (state.pointer_right != right)
            state.events.push_back({InputEvent::Type::PointerButton, Key::Tab, 1, right});
        state.pointer_right = right;
        state.look_active = right;
        if (action == AMOTION_EVENT_ACTION_SCROLL)
            state.scroll_y += GameActivityPointerAxes_getAxisValue(&event.pointers[0], AMOTION_EVENT_AXIS_VSCROLL);
        return;
    }

    const int32_t changed = (event.action & AMOTION_EVENT_ACTION_POINTER_INDEX_MASK) >> AMOTION_EVENT_ACTION_POINTER_INDEX_SHIFT;
    if (changed < 0 || static_cast<uint32_t>(changed) >= event.pointerCount) return;
    const auto& pointer = event.pointers[changed];
    const float x = GameActivityPointerAxes_getAxisValue(&pointer, AMOTION_EVENT_AXIS_X);
    const float y = GameActivityPointerAxes_getAxisValue(&pointer, AMOTION_EVENT_AXIS_Y);
    const bool down = action == AMOTION_EVENT_ACTION_DOWN || action == AMOTION_EVENT_ACTION_POINTER_DOWN;
    const bool up = action == AMOTION_EVENT_ACTION_UP || action == AMOTION_EVENT_ACTION_POINTER_UP;
    if (down) {
        if (y > height * 0.6f && x < width * 0.5f && move_pointer_ < 0) {
            move_pointer_ = pointer.id;
            move_origin_x_ = x;
            move_origin_y_ = y;
        } else if (y > height * 0.6f && x >= width * 0.5f && look_pointer_ < 0) {
            look_pointer_ = pointer.id;
            look_x_ = x;
            look_y_ = y;
            state.look_active = true;
        } else if (ui_pointer_ < 0) {
            ui_pointer_ = pointer.id;
            state.pointer_x = x;
            state.pointer_y = y;
            set_pointer(true);
        }
    }
    if (action == AMOTION_EVENT_ACTION_MOVE) {
        for (uint32_t i = 0; i < event.pointerCount; ++i) {
            const auto& moving = event.pointers[i];
            const float next_x = GameActivityPointerAxes_getAxisValue(&moving, AMOTION_EVENT_AXIS_X);
            const float next_y = GameActivityPointerAxes_getAxisValue(&moving, AMOTION_EVENT_AXIS_Y);
            if (moving.id == move_pointer_) {
                const float radius = 64.0f * std::max(scale, 1.0f);
                touch_move_x_ = std::clamp((next_x - move_origin_x_) / radius, -1.0f, 1.0f);
                touch_move_z_ = std::clamp((move_origin_y_ - next_y) / radius, -1.0f, 1.0f);
            } else if (moving.id == look_pointer_) {
                state.look_dx += (next_x - look_x_) / std::max(scale, 1.0f);
                state.look_dy += (next_y - look_y_) / std::max(scale, 1.0f);
                look_x_ = next_x;
                look_y_ = next_y;
            } else if (moving.id == ui_pointer_) {
                state.pointer_x = next_x;
                state.pointer_y = next_y;
            }
        }
    }
    if (up) {
        if (pointer.id == move_pointer_) { move_pointer_ = -1; touch_move_x_ = touch_move_z_ = 0; }
        if (pointer.id == look_pointer_) { look_pointer_ = -1; state.look_active = false; }
        if (pointer.id == ui_pointer_) {
            ui_pointer_ = -1;
            state.pointer_x = x;
            state.pointer_y = y;
            set_pointer(false);
        }
    }
}

void AndroidInput::read_text(void* context, const GameTextInputState* text) {
    auto& self = *static_cast<AndroidInput*>(context);
    if (!text) return;
    const std::string updated = text->text_UTF8 && text->text_length > 0
        ? std::string(text->text_UTF8, static_cast<size_t>(text->text_length)) : std::string{};
    size_t prefix = 0;
    while (prefix < updated.size() && prefix < self.ime_text_.size() && updated[prefix] == self.ime_text_[prefix]) ++prefix;
    // Start replacement at a UTF-8 character boundary.
    while (prefix > 0 && prefix < self.ime_text_.size() &&
           (static_cast<unsigned char>(self.ime_text_[prefix]) & 0xc0) == 0x80) --prefix;
    for (size_t i = prefix; i < self.ime_text_.size(); ++i) {
        if ((static_cast<unsigned char>(self.ime_text_[i]) & 0xc0) == 0x80) continue;
        self.state.events.push_back({InputEvent::Type::Key, Key::Backspace, 0, true});
        self.state.events.push_back({InputEvent::Type::Key, Key::Backspace, 0, false});
    }
    self.state.text.append(updated.substr(prefix));
    self.ime_text_ = updated;
}

void AndroidInput::begin_text_input(android_app& app) {
    // A new field must not inherit the previous field's IME edit history.
    ime_text_.clear();
    GameTextInputState empty{};
    empty.text_UTF8 = "";
    empty.selection = {0, 0};
    empty.composingRegion = {-1, -1};
    GameActivity_setTextInputState(app.activity, &empty);
    app.textInputState = 0;
}

void AndroidInput::process(android_app& app, float width, float height, float content_scale) {
    if (auto* buffer = android_app_swap_input_buffers(&app)) {
        for (uint64_t i = 0; i < buffer->keyEventsCount; ++i) key(buffer->keyEvents[i]);
        for (uint64_t i = 0; i < buffer->motionEventsCount; ++i) motion(buffer->motionEvents[i], width, height, content_scale);
        android_app_clear_key_events(buffer);
        android_app_clear_motion_events(buffer);
    }
    if (app.textInputState) {
        GameActivity_getTextInputState(app.activity, read_text, this);
        app.textInputState = 0;
    }
    state.move_x = std::clamp(touch_move_x_ + static_cast<float>(movement_keys_[3]) - static_cast<float>(movement_keys_[2]), -1.0f, 1.0f);
    state.move_z = std::clamp(touch_move_z_ + static_cast<float>(movement_keys_[0]) - static_cast<float>(movement_keys_[1]), -1.0f, 1.0f);
    state.move_y = static_cast<float>(movement_keys_[5]) - static_cast<float>(movement_keys_[4]);
}

void AndroidInput::release_all() {
    for (size_t i = 0; i < state.keys.size(); ++i) {
        if (state.keys[i]) state.events.push_back({InputEvent::Type::Key, static_cast<Key>(i), 0, false});
    }
    state.keys.fill(false);
    movement_keys_.fill(false);
    set_pointer(false);
    if (state.pointer_right) state.events.push_back({InputEvent::Type::PointerButton, Key::Tab, 1, false});
    state.pointer_right = state.look_active = false;
    state.move_x = state.move_y = state.move_z = 0;
    state.look_dx = state.look_dy = 0;
    touch_move_x_ = touch_move_z_ = 0;
    move_pointer_ = look_pointer_ = ui_pointer_ = -1;
}

void AndroidInput::clear_transients() {
    state.events.clear();
    state.text.clear();
    state.look_dx = state.look_dy = state.scroll_y = 0;
    state.toggle_ui = false;
}
}
