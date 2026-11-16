#include "android_input.hpp"
#include "android_platform.hpp"
#include <sandbox/viewer/viewer.hpp>
#include <android/log.h>
#include <android/looper.h>
#include <android/input.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <string>

#ifndef SANDBOX_ANDROID_TARGET_FPS
#define SANDBOX_ANDROID_TARGET_FPS 60
#endif
#ifndef SANDBOX_ANDROID_RENDER_SCALE
#define SANDBOX_ANDROID_RENDER_SCALE 1.0f
#endif

namespace sandbox::android {
namespace {
using Clock = std::chrono::steady_clock;
constexpr const char* log_tag = "VulkanSandbox";
struct SavedState {
    uint32_t signature = 0x56534b31; // versioned, application-owned CPU state only
    ViewState view;
};

class Runtime {
public:
    explicit Runtime(android_app& app) : app_(app), platform_(app) {
        if (app.savedState && app.savedStateSize == sizeof(SavedState)) {
            SavedState saved;
            std::memcpy(&saved, app.savedState, sizeof(saved));
            if (saved.signature == 0x56534b31 &&
                std::all_of(saved.view.position.begin(), saved.view.position.end(), [](float value) { return std::isfinite(value); }) &&
                std::isfinite(saved.view.yaw) && std::isfinite(saved.view.pitch)) view_ = saved.view;
        }
        app_.userData = this;
        app_.onAppCmd = command;
        // Admit pointer devices while leaving unrelated controller motion to the
        // system. Keep GameActivity's default key filter for volume/camera keys.
        android_app_set_motion_event_filter(&app_, [](const GameActivityMotionEvent* event) {
            return (event->source & AINPUT_SOURCE_CLASS_POINTER) != 0;
        });
        GameActivityPointerAxes_enableAxis(AMOTION_EVENT_AXIS_VSCROLL);
    }
    ~Runtime() { destroy_viewer(); }

    void run() {
        const auto interval = std::chrono::duration_cast<Clock::duration>(
            std::chrono::duration<double>(1.0 / std::clamp(SANDBOX_ANDROID_TARGET_FPS, 15, 240)));
        auto next_frame = Clock::now();
        auto previous_frame = next_frame;
        while (!app_.destroyRequested) {
            const bool active = can_render();
            int timeout_ms = -1;
            if (active) {
                const auto remaining = next_frame - Clock::now();
                timeout_ms = remaining > Clock::duration::zero()
                    ? static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(remaining).count() + 1) : 0;
            }
            int events = 0;
            android_poll_source* source = nullptr;
            int result = ALooper_pollOnce(timeout_ms, nullptr, &events, reinterpret_cast<void**>(&source));
            while (result >= 0) {
                if (source) source->process(&app_, source);
                if (app_.destroyRequested) break;
                source = nullptr;
                result = ALooper_pollOnce(0, nullptr, &events, reinterpret_cast<void**>(&source));
            }
            if (app_.destroyRequested) break;
            const auto extent = platform_.framebuffer_extent();
            input_.process(app_, static_cast<float>(extent.width), static_cast<float>(extent.height), platform_.content_scale());
            if (!can_render() || extent.width == 0 || extent.height == 0) {
                previous_frame = Clock::now();
                next_frame = previous_frame;
                continue;
            }
            if (Clock::now() < next_frame) continue;
            try {
                if (!viewer_) {
                    ViewerOptions options;
                    options.validation = true;
                    options.vsync = true;
                    options.render_scale = SANDBOX_ANDROID_RENDER_SCALE;
                    viewer_ = std::make_unique<Viewer>(platform_, view_, options);
                    previous_frame = Clock::now();
                    __android_log_print(ANDROID_LOG_INFO, log_tag, "Renderer created: %ux%u, density %.2f", extent.width, extent.height, platform_.content_scale());
                }
                const auto now = Clock::now();
                const float dt = std::clamp(std::chrono::duration<float>(now - previous_frame).count(), 0.001f, 0.1f);
                // Viewer retains transitions across skipped frames. Pass each
                // platform event exactly once, even when acquisition is deferred.
                viewer_->draw(input_.state, dt);
                input_.clear_transients();
                const bool wants_keyboard = viewer_->wants_text_input();
                if (wants_keyboard != keyboard_visible_) {
                    keyboard_visible_ = wants_keyboard;
                    if (wants_keyboard) {
                        input_.begin_text_input(app_);
                        GameActivity_showSoftInput(app_.activity, 0);
                    }
                    else GameActivity_hideSoftInput(app_.activity, 0);
                }
                previous_frame = now;
                next_frame = std::max(next_frame + interval, Clock::now());
            } catch (const std::exception& error) {
                fail(error.what());
            }
        }
        destroy_viewer();
        app_.userData = nullptr;
        app_.onAppCmd = nullptr;
    }

private:
    bool can_render() const {
        if (!resumed_ || !focused_ || !app_.window || failed_) return false;
        const auto extent = platform_.framebuffer_extent();
        return extent.width > 0 && extent.height > 0;
    }

    void destroy_viewer() noexcept {
        if (keyboard_visible_) {
            GameActivity_hideSoftInput(app_.activity, 0);
            keyboard_visible_ = false;
        }
        if (viewer_) {
            try {
                const auto errors = viewer_->shutdown();
                __android_log_print(errors ? ANDROID_LOG_ERROR : ANDROID_LOG_INFO, log_tag,
                    "Renderer released; validation errors: %u", errors);
            } catch (const std::exception& error) {
                __android_log_print(ANDROID_LOG_ERROR, log_tag, "Renderer shutdown: %s", error.what());
            }
            viewer_.reset();
        }
    }

    void fail(const char* reason) noexcept {
        failed_ = true;
        destroy_viewer();
        __android_log_print(ANDROID_LOG_ERROR, log_tag, "%s", reason);
        JNIEnv* env = nullptr;
        bool attached = false;
        if (app_.activity->vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6) == JNI_EDETACHED) {
            if (app_.activity->vm->AttachCurrentThread(&env, nullptr) != JNI_OK) {
                GameActivity_finish(app_.activity);
                return;
            }
            attached = true;
        }
        if (!env) return;
        auto activity = app_.activity->javaGameActivity;
        auto type = env->GetObjectClass(activity);
        auto method = env->GetMethodID(type, "showRendererError", "(Ljava/lang/String;)V");
        if (method) {
            auto message = env->NewStringUTF(reason);
            env->CallVoidMethod(activity, method, message);
            env->DeleteLocalRef(message);
        }
        if (env->ExceptionCheck()) {
            env->ExceptionDescribe();
            env->ExceptionClear();
            GameActivity_finish(app_.activity);
        }
        env->DeleteLocalRef(type);
        if (attached) app_.activity->vm->DetachCurrentThread();
    }

    static void command(android_app* app, int32_t command) noexcept {
        auto& runtime = *static_cast<Runtime*>(app->userData);
        switch (command) {
        case APP_CMD_RESUME:
            runtime.resumed_ = true;
            break;
        case APP_CMD_PAUSE:
        case APP_CMD_STOP:
            runtime.resumed_ = false;
            runtime.input_.release_all();
            if (runtime.keyboard_visible_) {
                GameActivity_hideSoftInput(app->activity, 0);
                runtime.keyboard_visible_ = false;
            }
            break;
        case APP_CMD_GAINED_FOCUS:
            runtime.focused_ = runtime.input_.state.focused = true;
            break;
        case APP_CMD_LOST_FOCUS:
            runtime.focused_ = runtime.input_.state.focused = false;
            runtime.input_.release_all();
            break;
        case APP_CMD_INIT_WINDOW:
        case APP_CMD_WINDOW_RESIZED:
        case APP_CMD_CONTENT_RECT_CHANGED:
            if (runtime.viewer_) runtime.viewer_->request_resize();
            break;
        case APP_CMD_CONFIG_CHANGED:
            // Recreate density-dependent UI resources while retaining camera state.
            runtime.destroy_viewer();
            runtime.input_.release_all();
            break;
        case APP_CMD_TERM_WINDOW:
            // This callback must finish before glue acknowledges window destruction.
            // The entire GPU session is disposable; application camera/UI state survives.
            runtime.destroy_viewer();
            runtime.input_.release_all();
            break;
        case APP_CMD_SAVE_STATE: {
            const SavedState saved{0x56534b31, runtime.view_};
            app->savedState = std::malloc(sizeof(saved));
            if (app->savedState) {
                std::memcpy(app->savedState, &saved, sizeof(saved));
                app->savedStateSize = sizeof(saved);
            }
            break;
        }
        case APP_CMD_DESTROY:
            runtime.destroy_viewer();
            break;
        default: break;
        }
    }

    android_app& app_;
    AndroidPlatform platform_;
    AndroidInput input_;
    ViewState view_;
    std::unique_ptr<Viewer> viewer_;
    bool resumed_ = false;
    bool focused_ = false;
    bool failed_ = false;
    bool keyboard_visible_ = false;
};
}
}

extern "C" void android_main(android_app* app) {
    sandbox::android::Runtime runtime(*app);
    runtime.run();
}
