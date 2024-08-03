#version 460

#extension GL_GOOGLE_include_directive : enable
#include"../include/host_device.hpp"

layout (location = 0) in vec3 fragPosWorld;
layout (location = 0) out vec4 FragColor;

layout (push_constant) uniform Push{PointsPC push;};

layout (early_fragment_tests) in;

void main()
{
    FragColor = push.pointsColor;
}
