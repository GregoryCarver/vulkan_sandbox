struct VertexInput
{
    [[vk::location(0)]] float3 position : POSITION;
    [[vk::location(1)]] float3 normal : NORMAL;
    [[vk::location(2)]] float2 uv : TEXCOORD0;
};

struct Camera
{
    column_major float4x4 view_projection;
};

struct Object
{
    column_major float4x4 model;
    float4 tint;
};

[[vk::binding(0, 0)]] ConstantBuffer<Camera> camera;
[[vk::binding(1, 0)]] Sampler2D checkerboard;
[[vk::push_constant]] ConstantBuffer<Object> object;

struct VertexOutput
{
    float4 position : SV_Position;
    float3 normal : NORMAL;
    float2 uv : TEXCOORD0;
};

[shader("vertex")]
VertexOutput vertMain(VertexInput input)
{
    VertexOutput output;
    float4 world = mul(object.model, float4(input.position, 1.0));
    output.position = mul(camera.view_projection, world);
    output.normal = mul((float3x3)object.model, input.normal);
    output.uv = input.uv;
    return output;
}

[shader("fragment")]
float4 fragMain(VertexOutput input) : SV_Target
{
    float3 normal = normalize(input.normal);
    float light = 0.3 + 0.7 * max(dot(normal, normalize(float3(0.5, 0.9, 0.6))), 0.0);
    float3 albedo = checkerboard.Sample(input.uv).rgb * object.tint.rgb;
    return float4(albedo * light, 1.0);
}
