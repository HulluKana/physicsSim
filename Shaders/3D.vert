#version 460

#extension GL_GOOGLE_include_directive : enable

#include"../include/host_device.hpp"

layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;

layout (location = 0) out vec3 fragPosWorld;
layout (location = 1) out vec3 fragNormalWorld;

layout(set = 0, binding = 0) uniform SceneUbo {Ubo ubo;};

void main()
{
    const vec4 worldPosition = vec4(position, 1.0);
    gl_Position = ubo.projViewMat * worldPosition;
    fragPosWorld = worldPosition.xyz;
    fragNormalWorld = normal;
}
