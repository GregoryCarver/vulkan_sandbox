#include <sandbox/viewer/scene.hpp>
#include <sandbox/gpu/resources.hpp>

#ifndef GLM_FORCE_RADIANS
#define GLM_FORCE_RADIANS
#endif
#ifndef GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#endif
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#if SANDBOX_ENABLE_IMGUI
#include <imgui.h>
#include <imgui_impl_vulkan.h>
#include "imgui_input.hpp"
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace sandbox {
namespace {
struct Vertex {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 uv;
};
struct CameraUniform { glm::mat4 view_projection; };
struct ObjectConstants {
    glm::mat4 model;
    glm::vec4 tint;
};
static_assert(sizeof(CameraUniform) == 64);
static_assert(sizeof(ObjectConstants) == 80);

struct CubeGeometry {
    std::array<Vertex, 24> vertices{};
    std::array<uint16_t, 36> indices{};
};

CubeGeometry make_cube(uint64_t revision) {
    const std::array<glm::vec3, 6> normals{{
        {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}}};
    const std::array<glm::vec3, 6> horizontal{{
        {0, 0, -1}, {0, 0, 1}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {-1, 0, 0}}};
    const std::array<glm::vec3, 6> vertical{{
        {0, 1, 0}, {0, 1, 0}, {0, 0, -1}, {0, 0, 1}, {0, 1, 0}, {0, 1, 0}}};
    constexpr std::array<glm::vec2, 4> corners{{{-1, -1}, {1, -1}, {1, 1}, {-1, 1}}};
    constexpr std::array<uint16_t, 6> face_indices{0, 1, 2, 2, 3, 0};
    CubeGeometry cube;
    const float height = 1.0f + 0.16f * std::sin(static_cast<float>(revision) * 1.3f);
    for (uint16_t face = 0; face < 6; ++face) {
        for (uint16_t corner = 0; corner < 4; ++corner) {
            auto position = (normals[face] + horizontal[face] * corners[corner].x
                             + vertical[face] * corners[corner].y) * 0.5f;
            position.y *= height;
            cube.vertices[face * 4 + corner] = {
                position, normals[face], corners[corner] * 0.5f + 0.5f};
        }
        for (uint16_t index = 0; index < 6; ++index)
            cube.indices[face * 6 + index] = static_cast<uint16_t>(face * 4 + face_indices[index]);
    }
    return cube;
}

std::vector<std::byte> make_checkerboard() {
    constexpr uint32_t side = 64;
    std::vector<std::byte> pixels(side * side * 4);
    for (uint32_t y = 0; y < side; ++y) {
        for (uint32_t x = 0; x < side; ++x) {
            const auto shade = static_cast<std::byte>(((x / 8) + (y / 8)) % 2 ? 200 : 255);
            const auto index = (y * side + x) * 4;
            pixels[index] = shade;
            pixels[index + 1] = shade;
            pixels[index + 2] = shade;
            pixels[index + 3] = std::byte{255};
        }
    }
    return pixels;
}

template<class T> auto bytes(const T& value) {
    return std::as_bytes(std::span(&value, 1));
}

#if SANDBOX_ENABLE_IMGUI
void check_imgui_vk(VkResult result) { check_vk(result, "ImGui Vulkan backend"); }
#endif
}

struct Scene::Impl {
    struct Mesh {
        Buffer vertices;
        Buffer indices;
        uint64_t revision;
        Mesh(Allocator& allocator, uint64_t version)
            : vertices(allocator, sizeof(CubeGeometry::vertices),
                       VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT),
              indices(allocator, sizeof(CubeGeometry::indices),
                      VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT),
              revision(version) {}
    };
    struct Frame {
        Buffer uniform;
        std::unique_ptr<Mesh> active;
        std::unique_ptr<Mesh> pending;
        uint64_t upload_ticket = 0;
        VkDescriptorSet descriptor = VK_NULL_HANDLE;
        explicit Frame(Allocator& allocator)
            : uniform(allocator, sizeof(CameraUniform), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                      MemoryUsage::Upload), active(std::make_unique<Mesh>(allocator, 0)) {}
    };

    Device& device;
    Allocator& allocator;
    UploadQueue& uploads;
    Platform& platform;
    ViewState& state;
    std::vector<Frame> frames;
    Image texture;
    Sampler sampler;
    VkDescriptorSetLayout descriptor_layout = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkFormat color_format = VK_FORMAT_UNDEFINED;
    VkFormat depth_format = VK_FORMAT_UNDEFINED;
    uint32_t image_count = 0;
    float elapsed = 0;
    float rotation = 0;
    float last_delta = 1.0f / 60.0f;
    uint64_t frame_count = 0;
    uint64_t mesh_revision = 0;
    float rebuild_timer = 0;
#if SANDBOX_ENABLE_IMGUI
    ImGuiContext* gui = nullptr;
    bool gui_renderer_ready = false;
    VkDescriptorPool gui_pool = VK_NULL_HANDLE;
    std::string ini_path;
    float ui_scale = 1.0f;
    VkExtent2D gui_extent{};
#endif

    Impl(Device& device_in, Allocator& allocator_in, UploadQueue& uploads_in,
         Platform& platform_in, ViewState& state_in, uint32_t frame_slots)
        : device(device_in), allocator(allocator_in), uploads(uploads_in),
          platform(platform_in), state(state_in),
          texture(allocator, ImageDesc{{64, 64}, VK_FORMAT_R8G8B8A8_SRGB,
                      VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT}),
          sampler(device, VK_FILTER_NEAREST, VK_SAMPLER_ADDRESS_MODE_REPEAT) {
        try {
            if (frame_slots == 0) throw std::invalid_argument("Scene requires a frame slot");
            frames.reserve(frame_slots);
            auto initial = uploads.begin();
            const auto cube = make_cube(0);
            for (uint32_t index = 0; index < frame_slots; ++index) {
                frames.emplace_back(allocator);
                upload_mesh(initial, *frames.back().active, cube);
            }
            const auto pixels = make_checkerboard();
            initial.upload_rgba8(texture, pixels);
            uploads.wait(uploads.submit(std::move(initial)));
            create_descriptors();
#if SANDBOX_ENABLE_IMGUI
            IMGUI_CHECKVERSION();
            gui = ImGui::CreateContext();
            ImGui::SetCurrentContext(gui);
            auto& io = ImGui::GetIO();
            io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
            io.BackendPlatformName = "sandbox_platform_input";
            ini_path = platform.writable_path("imgui.ini");
            io.IniFilename = ini_path.empty() ? nullptr : ini_path.c_str();
            ImGui::StyleColorsDark();
            auto& style = ImGui::GetStyle();
            style.WindowRounding = 6;
            style.FrameRounding = 3;
            style.Colors[ImGuiCol_WindowBg] = ImVec4(0.065f, 0.08f, 0.105f, 0.96f);
            VkDescriptorPoolSize gui_size{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 128};
            VkDescriptorPoolCreateInfo gui_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
            gui_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
            gui_info.maxSets = 128;
            gui_info.poolSizeCount = 1;
            gui_info.pPoolSizes = &gui_size;
            check_vk(vkCreateDescriptorPool(device.logical(), &gui_info, nullptr, &gui_pool),
                     "create ImGui descriptor pool");
#endif
        } catch (...) {
            destroy();
            throw;
        }
    }

    ~Impl() { destroy(); }

    void destroy() noexcept {
        // Shutdown and construction failures are infrequent, explicit idle points.
        // Ordinary frame updates and mesh uploads never wait the queue idle.
        if (device.logical()) vkDeviceWaitIdle(device.logical());
#if SANDBOX_ENABLE_IMGUI
        if (gui) {
            ImGui::SetCurrentContext(gui);
            // Init can throw via CheckVkResultFn after setting backend user data.
            if (ImGui::GetIO().BackendRendererUserData) ImGui_ImplVulkan_Shutdown();
            ImGui::DestroyContext(gui);
            gui = nullptr;
            gui_renderer_ready = false;
        }
        if (gui_pool) vkDestroyDescriptorPool(device.logical(), std::exchange(gui_pool, VK_NULL_HANDLE), nullptr);
#endif
        if (pipeline) vkDestroyPipeline(device.logical(), std::exchange(pipeline, VK_NULL_HANDLE), nullptr);
        if (pipeline_layout) vkDestroyPipelineLayout(device.logical(), std::exchange(pipeline_layout, VK_NULL_HANDLE), nullptr);
        if (descriptor_pool) vkDestroyDescriptorPool(device.logical(), std::exchange(descriptor_pool, VK_NULL_HANDLE), nullptr);
        if (descriptor_layout) vkDestroyDescriptorSetLayout(device.logical(), std::exchange(descriptor_layout, VK_NULL_HANDLE), nullptr);
    }

    static void upload_mesh(UploadBatch& batch, Mesh& mesh, const CubeGeometry& geometry) {
        batch.upload(mesh.vertices, bytes(geometry.vertices), 0,
                     VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT,
                     VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT);
        batch.upload(mesh.indices, bytes(geometry.indices), 0,
                     VK_PIPELINE_STAGE_2_INDEX_INPUT_BIT, VK_ACCESS_2_INDEX_READ_BIT);
    }

    void create_descriptors() {
        const std::array<VkDescriptorSetLayoutBinding, 2> bindings{{
            {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT, nullptr},
            {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr}}};
        VkDescriptorSetLayoutCreateInfo layout{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        layout.bindingCount = static_cast<uint32_t>(bindings.size());
        layout.pBindings = bindings.data();
        check_vk(vkCreateDescriptorSetLayout(device.logical(), &layout, nullptr, &descriptor_layout),
                 "create scene descriptor layout");
        const auto count = static_cast<uint32_t>(frames.size());
        const std::array<VkDescriptorPoolSize, 2> sizes{{
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, count},
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, count}}};
        VkDescriptorPoolCreateInfo pool{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pool.maxSets = count;
        pool.poolSizeCount = static_cast<uint32_t>(sizes.size());
        pool.pPoolSizes = sizes.data();
        check_vk(vkCreateDescriptorPool(device.logical(), &pool, nullptr, &descriptor_pool),
                 "create scene descriptor pool");
        const std::vector<VkDescriptorSetLayout> layouts(count, descriptor_layout);
        std::vector<VkDescriptorSet> sets(count);
        VkDescriptorSetAllocateInfo allocate{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        allocate.descriptorPool = descriptor_pool;
        allocate.descriptorSetCount = count;
        allocate.pSetLayouts = layouts.data();
        check_vk(vkAllocateDescriptorSets(device.logical(), &allocate, sets.data()),
                 "allocate scene descriptors");
        for (uint32_t index = 0; index < count; ++index) {
            auto& frame = frames[index];
            frame.descriptor = sets[index];
            const VkDescriptorBufferInfo buffer{frame.uniform.handle(), 0, sizeof(CameraUniform)};
            const VkDescriptorImageInfo image{sampler.handle(), texture.view(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            std::array<VkWriteDescriptorSet, 2> writes{};
            writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[0].dstSet = frame.descriptor;
            writes[0].dstBinding = 0;
            writes[0].descriptorCount = 1;
            writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            writes[0].pBufferInfo = &buffer;
            writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[1].dstSet = frame.descriptor;
            writes[1].dstBinding = 1;
            writes[1].descriptorCount = 1;
            writes[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writes[1].pImageInfo = &image;
            vkUpdateDescriptorSets(device.logical(), static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
        }
        const VkPushConstantRange constants{VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                                             0, sizeof(ObjectConstants)};
        VkPipelineLayoutCreateInfo pipeline_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pipeline_info.setLayoutCount = 1;
        pipeline_info.pSetLayouts = &descriptor_layout;
        pipeline_info.pushConstantRangeCount = 1;
        pipeline_info.pPushConstantRanges = &constants;
        check_vk(vkCreatePipelineLayout(device.logical(), &pipeline_info, nullptr, &pipeline_layout),
                 "create scene pipeline layout");
    }

    VkPipeline create_pipeline(VkFormat color, VkFormat depth) {
        const auto shader_bytes = platform.read_asset("shaders/cube.spv");
        if (shader_bytes.empty() || shader_bytes.size() % sizeof(uint32_t))
            throw std::runtime_error("shaders/cube.spv is not a complete SPIR-V module");
        std::vector<uint32_t> words(shader_bytes.size() / sizeof(uint32_t));
        std::memcpy(words.data(), shader_bytes.data(), shader_bytes.size());
        if (words[0] != 0x07230203) throw std::runtime_error("shaders/cube.spv has invalid SPIR-V magic");
        VkShaderModuleCreateInfo module_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        module_info.codeSize = shader_bytes.size();
        module_info.pCode = words.data();
        VkShaderModule module = VK_NULL_HANDLE;
        check_vk(vkCreateShaderModule(device.logical(), &module_info, nullptr, &module), "create cube shader");
        VkPipeline result = VK_NULL_HANDLE;
        try {
            std::array<VkPipelineShaderStageCreateInfo, 2> stages{};
            stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
            stages[0].module = module;
            stages[0].pName = "vertMain";
            stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
            stages[1].module = module;
            stages[1].pName = "fragMain";
            const VkVertexInputBindingDescription binding{0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX};
            const std::array<VkVertexInputAttributeDescription, 3> attributes{{
                {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, position)},
                {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal)},
                {2, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex, uv)}}};
            VkPipelineVertexInputStateCreateInfo vertex{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
            vertex.vertexBindingDescriptionCount = 1;
            vertex.pVertexBindingDescriptions = &binding;
            vertex.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributes.size());
            vertex.pVertexAttributeDescriptions = attributes.data();
            VkPipelineInputAssemblyStateCreateInfo assembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
            assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
            VkPipelineViewportStateCreateInfo viewport{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
            viewport.viewportCount = 1;
            viewport.scissorCount = 1;
            VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
            raster.polygonMode = VK_POLYGON_MODE_FILL;
            raster.cullMode = VK_CULL_MODE_BACK_BIT;
            raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
            raster.lineWidth = 1.0f;
            VkPipelineMultisampleStateCreateInfo samples{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
            samples.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
            VkPipelineDepthStencilStateCreateInfo depth_state{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
            depth_state.depthTestEnable = VK_TRUE;
            depth_state.depthWriteEnable = VK_TRUE;
            depth_state.depthCompareOp = VK_COMPARE_OP_LESS;
            VkPipelineColorBlendAttachmentState blend_attachment{};
            blend_attachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
                                             | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
            VkPipelineColorBlendStateCreateInfo blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
            blend.attachmentCount = 1;
            blend.pAttachments = &blend_attachment;
            const std::array<VkDynamicState, 2> dynamics{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
            VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
            dynamic.dynamicStateCount = static_cast<uint32_t>(dynamics.size());
            dynamic.pDynamicStates = dynamics.data();
            VkPipelineRenderingCreateInfo rendering{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
            rendering.colorAttachmentCount = 1;
            rendering.pColorAttachmentFormats = &color;
            rendering.depthAttachmentFormat = depth;
            VkGraphicsPipelineCreateInfo info{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
            info.pNext = &rendering;
            info.stageCount = static_cast<uint32_t>(stages.size());
            info.pStages = stages.data();
            info.pVertexInputState = &vertex;
            info.pInputAssemblyState = &assembly;
            info.pViewportState = &viewport;
            info.pRasterizationState = &raster;
            info.pMultisampleState = &samples;
            info.pDepthStencilState = &depth_state;
            info.pColorBlendState = &blend;
            info.pDynamicState = &dynamic;
            info.layout = pipeline_layout;
            check_vk(vkCreateGraphicsPipelines(device.logical(), VK_NULL_HANDLE, 1, &info, nullptr, &result),
                     "create textured cube pipeline");
        } catch (...) {
            if (result) vkDestroyPipeline(device.logical(), result, nullptr);
            vkDestroyShaderModule(device.logical(), module, nullptr);
            throw;
        }
        vkDestroyShaderModule(device.logical(), module, nullptr);
        return result;
    }

    void change_swapchain(VkFormat color, VkFormat depth, uint32_t count) {
#if SANDBOX_ENABLE_IMGUI
        if (count < 2) throw std::runtime_error("The sample requires at least two swapchain images");
#else
        if (count == 0) throw std::runtime_error("The sample requires a swapchain image");
#endif
        const bool formats_changed = color_format != color || depth_format != depth;
        if (!pipeline || formats_changed) {
            const auto replacement = create_pipeline(color, depth);
            if (pipeline) vkDestroyPipeline(device.logical(), pipeline, nullptr);
            pipeline = replacement;
        }
#if SANDBOX_ENABLE_IMGUI
        ImGui::SetCurrentContext(gui);
        if (!gui_renderer_ready || formats_changed || image_count != count) {
            if (gui_renderer_ready) {
                ImGui_ImplVulkan_Shutdown();
                gui_renderer_ready = false;
            }
            ImGui_ImplVulkan_InitInfo info{};
            info.ApiVersion = VK_API_VERSION_1_3;
            info.Instance = device.instance();
            info.PhysicalDevice = device.physical();
            info.Device = device.logical();
            info.QueueFamily = device.graphics_family();
            info.Queue = device.graphics_queue();
            info.DescriptorPool = gui_pool;
            info.MinImageCount = 2;
            info.ImageCount = count;
            info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
            info.UseDynamicRendering = true;
            info.PipelineRenderingCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
            info.PipelineRenderingCreateInfo.colorAttachmentCount = 1;
            info.PipelineRenderingCreateInfo.pColorAttachmentFormats = &color;
            info.CheckVkResultFn = check_imgui_vk;
            gui_renderer_ready = ImGui_ImplVulkan_Init(&info);
            if (!gui_renderer_ready) throw std::runtime_error("ImGui Vulkan initialization failed");
        }
#endif
        color_format = color;
        depth_format = depth;
        image_count = count;
    }

    void refresh_mesh(Frame& frame) {
        if (frame.pending && uploads.is_complete(frame.upload_ticket)) {
            // The owner already waited this slot's draw fence before prepare.
            // Both the old draw and the replacement upload are now complete.
            frame.active = std::move(frame.pending);
        }
        if (!frame.pending && frame.active->revision != mesh_revision) {
            frame.pending = std::make_unique<Mesh>(allocator, mesh_revision);
            auto batch = uploads.begin();
            upload_mesh(batch, *frame.pending, make_cube(mesh_revision));
            frame.upload_ticket = uploads.submit(std::move(batch));
        }
    }

    void update_camera(const InputState& input, float dt, bool capture_mouse, bool capture_keyboard) {
        if (!input.focused) return;
        if (input.look_active && !capture_mouse) {
            state.yaw += input.look_dx * 0.12f;
            state.pitch = std::clamp(state.pitch - input.look_dy * 0.12f, -88.0f, 88.0f);
        }
        if (capture_keyboard || (input.pointer_down && capture_mouse)) return;
        const float yaw = glm::radians(state.yaw);
        const float pitch = glm::radians(state.pitch);
        const glm::vec3 forward{std::cos(yaw) * std::cos(pitch), std::sin(pitch),
                                std::sin(yaw) * std::cos(pitch)};
        const glm::vec3 right = glm::normalize(glm::cross(forward, glm::vec3(0, 1, 0)));
        glm::vec3 direction = right * input.move_x + glm::vec3(0, 1, 0) * input.move_y + forward * input.move_z;
        if (glm::length(direction) > 1.0f) direction = glm::normalize(direction);
        auto position = glm::vec3(state.position[0], state.position[1], state.position[2]);
        position += direction * dt * 3.0f;
        state.position = {position.x, position.y, position.z};
    }

#if SANDBOX_ENABLE_IMGUI
    void begin_gui(const InputState& input, float dt, VkExtent2D extent) {
        ImGui::SetCurrentContext(gui);
        const float scale = std::clamp(platform.content_scale(), 0.75f, 4.0f);
        if (scale != ui_scale) {
            auto& style = ImGui::GetStyle();
            style.ScaleAllSizes(scale / ui_scale);
            style.FontScaleDpi = scale;
            ui_scale = scale;
        }
        gui_extent = extent;
        auto& io = ImGui::GetIO();
        io.DisplaySize = ImVec2(static_cast<float>(extent.width), static_cast<float>(extent.height));
        io.DisplayFramebufferScale = ImVec2(1, 1);
        io.DeltaTime = dt;
        detail::feed_imgui_input(io, input);
        ImGui_ImplVulkan_NewFrame();
        ImGui::NewFrame();
    }

    void draw_gui() {
        if (!state.show_ui) return;
        const float margin = 18 * ui_scale;
        const float width = std::max(1.0f, std::min(345 * ui_scale, static_cast<float>(gui_extent.width) - margin * 2));
        const float height = std::max(1.0f, static_cast<float>(gui_extent.height) - margin * 2);
        ImGui::SetNextWindowPos(ImVec2(margin, margin), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(width, 0), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSizeConstraints(ImVec2(std::min(240 * ui_scale, width), std::min(80 * ui_scale, height)),
                                          ImVec2(width, height));
        if (ImGui::Begin("Vulkan Sandbox")) {
            ImGui::TextUnformatted("WINDOWS + ANDROID / SHARED SCENE");
            ImGui::Separator();
            ImGui::TextWrapped("%s", device.properties().deviceName);
            ImGui::Text("%.1f ms  |  %.0f FPS", last_delta * 1000.0f, 1.0f / last_delta);
            ImGui::Text("Frame slots: %u  |  Swapchain images: %u",
                        static_cast<unsigned>(frames.size()), image_count);
            ImGui::Text("Validation errors: %u", device.validation_errors());
            ImGui::Separator();
            ImGui::Checkbox("Animate cubes", &state.animate);
            ImGui::Checkbox("Show ImGui demo", &state.show_demo);
            if (ImGui::Button("Rebuild mesh")) ++mesh_revision;
            ImGui::SameLine();
            ImGui::Checkbox("Stream mesh updates", &state.stream_mesh_updates);
            ImGui::Text("Mesh revision: %llu  |  Uploads pending: %zu",
                        static_cast<unsigned long long>(mesh_revision), uploads.pending_count());
            ImGui::TextWrapped("Rebuilds upload new buffers while previous frames keep their own mesh alive.");
            if (ImGui::Button("Reset camera")) {
                state.position = {0.0f, 1.0f, 4.0f};
                state.yaw = -90;
                state.pitch = -12;
            }
            ImGui::Separator();
            ImGui::TextWrapped("Desktop: hold right mouse to look; WASD to move; Q/E down/up; F1 toggles UI.");
            ImGui::TextWrapped("Touch: use the bottom 40%% to move (left) and look (right). Use the upper area for UI.");
        }
        ImGui::End();
        if (state.show_demo) {
            const float demo_width = std::max(1.0f, std::min(410 * ui_scale, static_cast<float>(gui_extent.width) - margin * 2));
            ImGui::SetNextWindowPos(ImVec2(std::max(margin, static_cast<float>(gui_extent.width) - demo_width - margin), margin),
                                    ImGuiCond_FirstUseEver);
            ImGui::SetNextWindowSize(ImVec2(demo_width, std::min(580 * ui_scale, height)), ImGuiCond_FirstUseEver);
            // Keep the scene accessible on narrow displays; the demo remains
            // available through its title bar and the shared debug panel.
            ImGui::SetNextWindowCollapsed(gui_extent.width < 1000 * ui_scale, ImGuiCond_FirstUseEver);
            ImGui::ShowDemoWindow(&state.show_demo);
        }
    }
#endif

    void prepare(uint32_t frame_index, const InputState& input, float delta, VkExtent2D extent) {
        if (frame_index >= frames.size()) throw std::out_of_range("Scene frame slot");
        if (!pipeline || !extent.width || !extent.height)
            throw std::logic_error("Scene must have a nonzero swapchain before preparing a frame");
        const float dt = std::clamp(delta, 0.001f, 0.1f);
        last_delta += (dt - last_delta) * 0.1f;
        elapsed += dt;
        if (state.animate) rotation += dt * 0.3f;
        ++frame_count;
        if (input.toggle_ui) state.show_ui = !state.show_ui;
        bool capture_mouse = false;
        bool capture_keyboard = false;
#if SANDBOX_ENABLE_IMGUI
        begin_gui(input, dt, extent);
        capture_mouse = state.show_ui && ImGui::GetIO().WantCaptureMouse;
        capture_keyboard = state.show_ui && ImGui::GetIO().WantCaptureKeyboard;
#endif
        update_camera(input, dt, capture_mouse, capture_keyboard);
#if SANDBOX_ENABLE_IMGUI
        draw_gui();
        ImGui::Render();
#endif
        if (state.stream_mesh_updates) {
            rebuild_timer += dt;
            if (rebuild_timer >= 0.25f) { ++mesh_revision; rebuild_timer = 0; }
        }
        auto& frame = frames[frame_index];
        refresh_mesh(frame);
        const auto position = glm::vec3(state.position[0], state.position[1], state.position[2]);
        const auto yaw = glm::radians(state.yaw);
        const auto pitch = glm::radians(state.pitch);
        const glm::vec3 forward{std::cos(yaw) * std::cos(pitch), std::sin(pitch),
                                std::sin(yaw) * std::cos(pitch)};
        const auto view = glm::lookAtRH(position, position + forward, glm::vec3(0, 1, 0));
        auto projection = glm::perspectiveRH_ZO(glm::radians(60.0f),
            static_cast<float>(extent.width) / static_cast<float>(extent.height), 0.05f, 100.0f);
        projection[1][1] *= -1;
        const CameraUniform uniform{projection * view};
        frame.uniform.write(bytes(uniform));
    }

    void record(VkCommandBuffer command, uint32_t index, VkExtent2D extent,
                VkFormat color, VkFormat depth, uint32_t count) {
        if (color != color_format || depth != depth_format || count != image_count)
            throw std::logic_error("Call Scene::on_swapchain_changed while idle before recording");
        const auto& frame = frames.at(index);
        vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
        const VkViewport viewport{0, 0, static_cast<float>(extent.width), static_cast<float>(extent.height), 0, 1};
        const VkRect2D scissor{{0, 0}, extent};
        vkCmdSetViewport(command, 0, 1, &viewport);
        vkCmdSetScissor(command, 0, 1, &scissor);
        const VkBuffer vertex = frame.active->vertices.handle();
        const VkDeviceSize offset = 0;
        vkCmdBindVertexBuffers(command, 0, 1, &vertex, &offset);
        vkCmdBindIndexBuffer(command, frame.active->indices.handle(), 0, VK_INDEX_TYPE_UINT16);
        vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
                                0, 1, &frame.descriptor, 0, nullptr);
        const auto draw_cube = [&](glm::vec3 position, glm::vec3 scale, float angle, glm::vec4 tint) {
            auto model = glm::translate(glm::mat4(1), position);
            model = glm::rotate(model, angle, glm::vec3(0, 1, 0));
            model = glm::scale(model, scale);
            const ObjectConstants object{model, tint};
            vkCmdPushConstants(command, pipeline_layout,
                VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(object), &object);
            vkCmdDrawIndexed(command, 36, 1, 0, 0, 0);
        };
        draw_cube({0, 0, 0}, {1, 1, 1}, rotation, {1.0f, 0.59f, 0.22f, 1});
        draw_cube({-1.6f, -0.15f, -1.2f}, {0.7f, 0.7f, 0.7f}, -rotation * 0.65f, {0.2f, 0.65f, 1.0f, 1});
        draw_cube({1.5f, 0.05f, -0.8f}, {1.1f, 1.1f, 1.1f}, rotation * 0.5f, {0.25f, 0.85f, 0.55f, 1});
        draw_cube({0, -0.9f, -0.6f}, {7.0f, 0.2f, 6.0f}, 0, {0.21f, 0.28f, 0.35f, 1});
    }

    void record_ui(VkCommandBuffer command) {
#if SANDBOX_ENABLE_IMGUI
        ImGui::SetCurrentContext(gui);
        ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), command);
#else
        (void)command;
#endif
    }
};

Scene::Scene(Device& device, Allocator& allocator, UploadQueue& uploads,
             Platform& platform, ViewState& state, uint32_t frames)
    : impl_(std::make_unique<Impl>(device, allocator, uploads, platform, state, frames)) {}
Scene::~Scene() = default;
void Scene::on_swapchain_changed(VkFormat color, VkFormat depth, uint32_t count) {
    impl_->change_swapchain(color, depth, count);
}
void Scene::prepare(uint32_t frame, const InputState& input, float delta, VkExtent2D extent) {
    impl_->prepare(frame, input, delta, extent);
}
void Scene::record(VkCommandBuffer command, uint32_t frame, VkExtent2D extent,
                   VkFormat color, VkFormat depth, uint32_t count) {
    impl_->record(command, frame, extent, color, depth, count);
}
void Scene::record_ui(VkCommandBuffer command) { impl_->record_ui(command); }
bool Scene::wants_text_input() const {
#if SANDBOX_ENABLE_IMGUI
    ImGui::SetCurrentContext(impl_->gui);
    return ImGui::GetIO().WantTextInput;
#else
    return false;
#endif
}
}
