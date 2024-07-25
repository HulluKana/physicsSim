#ifndef HOST_DEVICE
#define HOST_DEVICE

#ifdef __cplusplus

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include<glm/glm.hpp>

using uint = uint32_t;
using vec4 = glm::vec4;
using mat4 = glm::mat4;
#endif

struct Ubo {
    mat4 projViewMat;
    vec4 camPos;
    vec4 lightPos;
    vec4 lightColor;
    vec4 ambientLightColor;
};

struct DefaultPC {
    uint matIdx;
};

struct WireframePC {
    vec4 wireframeColor;
};

struct NormalUpdaterPC {
    uint triangleCount;
    uint vertexOffset;
    uint indexOffset;
};

#endif
