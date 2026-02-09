#version 450
#extension GL_EXT_nonuniform_qualifier : enable

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 fragTexCoord;
layout(location = 2) in vec3 fragWorldPos;
layout(location = 3) in vec3 fragNormal;

layout(set = 1, binding = 0) uniform sampler texSampler;
layout(set = 1, binding = 1) uniform texture2D textures[];

layout(push_constant) uniform PushConstants {
    mat4 model;
    uint textureId;
    uint normalId;
    uint emissiveId;
    float alphaCutoff;
    uint debugFlags;
} pcs;

layout(location = 0) out vec4 outColor;
layout(location = 1) out vec4 outEmissive;
layout(location = 2) out vec4 outNormal;

const uint DEBUG_COLOR_NORMALS = 1u << 0;
const uint DEBUG_COLOR_DEPTH = 1u << 1;
const uint DEBUG_COLOR_WIREFRAME = 1u << 2;

void main() {
    vec4 pixelColor;
    if ((pcs.debugFlags & DEBUG_COLOR_WIREFRAME) != 0u) {
        pixelColor = vec4(vec3(0.0, 1.0, 1.0), 1.0);
    }
    else {
        pixelColor = texture(sampler2D(textures[pcs.textureId], texSampler), fragTexCoord);
    }
    if (pixelColor.a < pcs.alphaCutoff) discard;
    outColor = pixelColor;
    if (pcs.normalId != 0) {
        outNormal = texture(sampler2D(textures[pcs.normalId], texSampler), fragTexCoord);
    } else {
        outNormal = vec4(fragNormal, 1.0);
    }
}
