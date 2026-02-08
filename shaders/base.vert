#version 450

layout(binding = 0) uniform UniformBufferObject {
    mat4 view;
    mat4 proj;
} ubo;

layout(push_constant) uniform PushConstants {
    mat4 model;
    uint textureId;
    vec4 color;
    float brightness;
    uint debugFlags;
} pcs;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inColor;
layout(location = 3) in vec2 inTexCoord;

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec2 fragTexCoord;
layout(location = 2) out vec3 fragWorldPos;
layout(location = 3) out vec3 fragNormal;


void main() {
    vec4 worldPos = pcs.model * vec4(inPosition, 1.0);
    gl_Position = ubo.proj * ubo.view * worldPos;
    fragColor = inColor;
    fragTexCoord = inTexCoord;

    fragWorldPos = worldPos.xyz;
    mat3 matMult = transpose(inverse(mat3(pcs.model)));
    if (inNormal == vec3(0.0, 0.0, 0.0)) {
        fragNormal = inNormal;
    } else {
        fragNormal = normalize(matMult * inNormal);
    }
}
