#include <rendering.hpp>
#include <simulation.hpp>
#include <imgui.h>
#include <memory>
#include <vulkano_program.hpp>
#include <iostream>

int main()
{
    vul::Vulkano vulkano(2560, 1440, "Physics simulator");
    vul::settings::maxFps = 144.0;
    RenderingResources renderingResources = initializeRenderingStuff("../Models", "Physic sim scene.gltf", vulkano);
    renderingResources.simsPerFrame = 1000;
    Obj origObj = getObjFromScene(vulkano.scene, "Cube");
    Obj obj = origObj;
    while (true) {
        RenderResult renderingResult = render(vulkano, renderingResources, obj, origObj.energies);
        if (renderingResult.exit) break;
        if (renderingResult.skipFrame) continue;
        if (renderingResult.reset) obj = origObj;
        simulate(obj, renderingResult.frameTime, renderingResult.simDeltaT);
    }
    vulkano.letVulkanoFinish();
}
