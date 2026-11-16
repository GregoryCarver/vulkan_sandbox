#pragma once
#include <sandbox/viewer/input.hpp>
#include <game-activity/native_app_glue/android_native_app_glue.h>
#include <array>
#include <string>

namespace sandbox::android {
// Translates Android events into the same application input used on Windows.
// Camera gestures use the bottom 40% of the surface; the rest remains UI input.
class AndroidInput {
public:
    void process(android_app& app, float width, float height, float content_scale);
    void begin_text_input(android_app& app);
    void release_all();
    void clear_transients();
    InputState state;
private:
    void key(const GameActivityKeyEvent& event);
    void motion(const GameActivityMotionEvent& event, float width, float height, float content_scale);
    void set_pointer(bool down);
    static void read_text(void* context, const GameTextInputState* text);
    std::array<bool, 8> movement_keys_{};
    int32_t move_pointer_ = -1, look_pointer_ = -1, ui_pointer_ = -1;
    float move_origin_x_ = 0, move_origin_y_ = 0, look_x_ = 0, look_y_ = 0;
    float touch_move_x_ = 0, touch_move_z_ = 0;
    std::string ime_text_;
};
}
