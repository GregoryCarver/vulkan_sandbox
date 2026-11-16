#include "imgui_input.hpp"
#include <iostream>
#include <stdexcept>

namespace {
void expect(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

struct FrameResult {
    bool button_activated;
    bool key_pressed;
    bool key_released;
};

FrameResult draw_frame(const sandbox::InputState& input) {
    sandbox::detail::feed_imgui_input(ImGui::GetIO(), input);
    ImGui::NewFrame();
    ImGui::SetNextWindowPos(ImVec2(0, 0));
    ImGui::SetNextWindowSize(ImVec2(250, 150));
    ImGui::Begin("Input regression", nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize);
    ImGui::SetCursorScreenPos(ImVec2(20, 20));
    const bool clicked = ImGui::Button("Click", ImVec2(100, 40));
    const FrameResult result{clicked, ImGui::IsKeyPressed(ImGuiKey_A, false), ImGui::IsKeyReleased(ImGuiKey_A)};
    ImGui::End();
    ImGui::Render();
    return result;
}
}

int main() {
    ImGui::CreateContext();
    try {
        auto& io = ImGui::GetIO();
        io.DisplaySize = ImVec2(640, 480);
        io.DeltaTime = 1.0f / 60.0f;
        io.IniFilename = nullptr;
        unsigned char* pixels = nullptr;
        int width = 0, height = 0;
        io.Fonts->GetTexDataAsRGBA32(&pixels, &width, &height);
        io.Fonts->SetTexID(static_cast<ImTextureID>(1));

        using sandbox::InputEvent;
        using sandbox::Key;
        sandbox::InputState input;
        input.pointer_x = 50;
        input.pointer_y = 40;
        draw_frame(input); // Establish the button as an input target.
        draw_frame(input);

        // A complete click arrives between rendering polls, so the final held
        // snapshot is false. The widget must still activate once, on release.
        input.events = {{InputEvent::Type::PointerButton, Key::Tab, 0, true},
                        {InputEvent::Type::PointerButton, Key::Tab, 0, false}};
        auto click = draw_frame(input);
        expect(!click.button_activated, "Button must wait for the queued release");
        input.events.clear();
        click = draw_frame(input);
        expect(click.button_activated, "Fast press+release must activate the real ImGui button");
        expect(!draw_frame(input).button_activated, "Held-state snapshots must not duplicate clicks");

        input.events = {{InputEvent::Type::Key, Key::A, 0, true},
                        {InputEvent::Type::Key, Key::A, 0, false}};
        auto key = draw_frame(input);
        expect(key.key_pressed, "Fast keyboard tap must preserve its press");
        input.events.clear();
        key = draw_frame(input);
        expect(key.key_released, "Fast keyboard tap must preserve its release");
        expect(!draw_frame(input).key_pressed, "Held-state snapshots must not duplicate key taps");

        ImGui::DestroyContext();
        std::cout << "ImGui input regression checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        ImGui::DestroyContext();
        std::cerr << error.what() << '\n';
        return 1;
    }
}
