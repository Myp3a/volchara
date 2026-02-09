#version 450
#extension GL_EXT_nonuniform_qualifier : enable

#define ALLOW_NO_NORMAL

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 fragTexCoord;
layout(location = 2) in vec3 fragWorldPos;
layout(location = 3) in vec3 fragNormal;

layout(set=0, binding=0) uniform UniformBufferObject {
    mat4 view;
    mat4 proj;
} ubo;

layout(set = 1, binding = 0) uniform sampler texSampler;
layout(set = 1, binding = 1) uniform texture2D textures[];

struct SingleLight {
    vec4 position;
    vec4 color;
};

struct LightsHeader {
    vec4 ambient;
    uint lightCount;
    vec3 pad;
};

layout(set=2, binding=0) buffer LightsSSBO {
    LightsHeader header;
    SingleLight lights[];
} ssbo;

layout(push_constant) uniform PushConstants {
    mat4 model;
    uint textureId;
    uint normalId;
    uint emissiveId;
    float alphaCutoff;
    uint debugFlags;
} pcs;

layout(location = 0) out vec4 outColor;

const uint DEBUG_COLOR_NORMALS = 1u << 0;
const uint DEBUG_COLOR_DEPTH = 1u << 1;
const uint DEBUG_COLOR_WIREFRAME = 1u << 2;
const uint DEBUG_COLOR_UNLIT = 1u << 3;

float calcLightIntensity(vec3 lightPos, float lightBrightness, vec3 fragPos, vec3 fragNormal, bool physical) {
    vec3 lightVec = lightPos - fragPos;
    float lightDist = length(lightVec);
    float lightDistanceIntensity = 0.0;
    if (physical) {
        float lightAttenuation = lightBrightness / max(lightDist*lightDist, 1e-4);
        lightDistanceIntensity = min(max(lightAttenuation, 0.0), 1.0);
    }
    else {
        lightDistanceIntensity = min(max(lightBrightness - lightDist, 0.0), 1.0);
    }
    #ifdef ALLOW_NO_NORMAL
        if (fragNormal == vec3(0.0, 0.0, 0.0)) {
            return lightDistanceIntensity;
        }
    #endif
    float lightNormalizedIntensity = max(dot(fragNormal, normalize(lightVec)), 0.0);
    return lightNormalizedIntensity * lightDistanceIntensity;
}

void main() {
    vec4 inColorAndAlpha = texture(sampler2D(textures[pcs.textureId], texSampler), fragTexCoord);
    vec3 inColor = inColorAndAlpha.xyz;
    float inAlpha = inColorAndAlpha.a;
    vec3 inNormal;
    if (pcs.normalId != 0) {
        inNormal = texture(sampler2D(textures[pcs.normalId], texSampler), fragTexCoord).xyz;
    } else {
        inNormal = fragNormal;
    }
    
    if ((pcs.debugFlags & DEBUG_COLOR_NORMALS) != 0u) {
        outColor = vec4(abs(inNormal), 1.0);
        return;
    }
    if ((pcs.debugFlags & DEBUG_COLOR_DEPTH) != 0u) {
        outColor = vec4(vec3(0.0, gl_FragCoord.z, gl_FragCoord.z) * inAlpha, inAlpha);
        return;
    }
    if ((pcs.debugFlags & DEBUG_COLOR_WIREFRAME) != 0u) {
        outColor = vec4(vec3(0.0, 1.0, 1.0), 1.0);
        return;
    }
    if ((pcs.debugFlags & DEBUG_COLOR_UNLIT) != 0u) {
        outColor = vec4(inColor * inAlpha, inAlpha);
        return;
    }

    vec3 finalColor = vec3(0.0, 0.0, 0.0);
    if (pcs.emissiveId != 0) {
        finalColor = texture(sampler2D(textures[pcs.emissiveId], texSampler), fragTexCoord).xyz;
    } else {
        for (int i = 0; i < ssbo.header.lightCount; i++) {
            vec3 lightPos = ssbo.lights[i].position.xyz;
            float lightBrightness = ssbo.lights[i].color.w;
            float lightIntensity = calcLightIntensity(lightPos, lightBrightness, fragWorldPos, inNormal, false);
            finalColor += ssbo.lights[i].color.xyz * lightIntensity * inColor;
        }
        finalColor += ssbo.header.ambient.xyz * ssbo.header.ambient.w * inColor;
    }
    outColor = vec4(finalColor * inAlpha, inAlpha);
}