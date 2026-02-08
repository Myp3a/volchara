#version 450

#define ALLOW_NO_NORMAL

layout(input_attachment_index=0, set=0, binding=0) uniform subpassInput spColor;
layout(input_attachment_index=1, set=0, binding=1) uniform subpassInput spNormal;
layout(input_attachment_index=2, set=0, binding=2) uniform subpassInput spDepth;

layout(location=0) in vec2 inNDC;

layout(set=1, binding=0) uniform UniformBufferObject {
    mat4 view;
    mat4 proj;
} ubo;

layout(push_constant) uniform PushConstants {
    mat4 model;
    uint textureId;
    uint normalId;
    vec4 color;
    float brightness;
    uint debugFlags;
} pcs;

layout(location = 0) out vec4 outColor;

const uint DEBUG_COLOR_NORMALS = 1u << 0;
const uint DEBUG_COLOR_DEPTH = 1u << 1;
const uint DEBUG_COLOR_WIREFRAME = 1u << 2;

vec3 reconstructFragWorldPos(float depth, vec2 ndc) {
    mat4 invViewProj = inverse(ubo.proj * ubo.view);
    vec4 clip = vec4(ndc, depth, 1.0);
    vec4 world = invViewProj * clip;
    return world.xyz / world.w;
}

vec3 reconstructLightWorldPos(mat4 model) {
    return vec3(model * vec4(0,0,0,1));
}

float calcLightIntensity(vec3 lightPos, vec3 fragPos, vec3 fragNormal, bool physical) {
    vec3 lightVec = lightPos - fragPos;
    float lightDist = length(lightVec);
    float lightDistanceIntensity = 0.0;
    if (physical) {
        float lightAttenuation = pcs.brightness / max(lightDist*lightDist, 1e-4);
        lightDistanceIntensity = min(max(lightAttenuation, 0.0), 1.0);
    }
    else {
        lightDistanceIntensity = min(max(pcs.brightness - lightDist, 0.0), 1.0);
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
    vec3 inColor = subpassLoad(spColor).xyz;
    vec3 inNormal = subpassLoad(spNormal).xyz;
    float inDepth = subpassLoad(spDepth).x;
    
    if ((pcs.debugFlags & DEBUG_COLOR_NORMALS) != 0u) {
        outColor = vec4(abs(inNormal), 1.0);
        return;
    }
    if ((pcs.debugFlags & DEBUG_COLOR_DEPTH) != 0u) {
        outColor = vec4(vec3(inDepth), 1.0);
        return;
    }
    if ((pcs.debugFlags & DEBUG_COLOR_WIREFRAME) != 0u) {
        outColor = vec4(inColor, 1.0);
        return;
    }
    #ifdef IGNORE_LIGHTS
        outColor = vec4(inColor, 1.0);
        return;
    #endif

    if (pcs.color.w == 1.0) {
        vec3 ambientColor;
        if (pcs.color.r != 0 || pcs.color.g != 0 || pcs.color.b != 0) {
            ambientColor = pcs.color.xyz * pcs.brightness;
        }
        else {
            ambientColor = vec3(1.0, 1.0, 1.0);
        }
        outColor = vec4(ambientColor * inColor, 1.0);
    }

    else {
        vec3 lightPos = reconstructLightWorldPos(pcs.model);
        vec3 fragWorldPos = reconstructFragWorldPos(inDepth, inNDC);
        float lightIntensity = calcLightIntensity(lightPos, fragWorldPos, inNormal, false);
        vec3 lightColor = pcs.color.xyz * lightIntensity * inColor;
        outColor = vec4(lightColor, 1.0);
    }
}