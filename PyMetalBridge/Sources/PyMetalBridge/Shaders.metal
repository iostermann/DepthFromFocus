// https://developer.apple.com/documentation/metal/libraries/building_a_library_with_metal_s_command-line_tools
#include <metal_stdlib>
using namespace metal;

// Adapted from https://metalbyexample.com/fundamentals-of-image-processing/
// Needs to be adapted to use a 2d buffer instead
kernel void ComputeFocusMetric(texture2d<float, access::read> inTexture [[texture(0)]],
                       texture2d<float, access::write> outTexture [[texture(1)]],
                       texture2d<float, access::read> weights [[texture(2)]],
                       uint2 gid [[thread_position_in_grid]]) {
    int size = weights.get_width();
    int radius = size / 2;
     
    float4 accumColor(0, 0, 0, 0);
    for (int j = 0; j < size; ++j) {
        for (int i = 0; i < size; ++i) {
            uint2 kernelIndex(i, j);
            uint2 textureIndex(gid.x + (i - radius), gid.y + (j - radius));
            float4 color = inTexture.read(textureIndex).rgba;
            float4 weight = weights.read(kernelIndex).rrrr;
            accumColor += weight * color;
        }
    }
     
    outTexture.write(float4(accumColor.rgb, 1), gid);
}

kernel void ComputeFocusMetricFlat(const device float *inVector [[ buffer(0) ]],
                    device float *outVector [[ buffer(1) ]],
                    uint id [[ thread_position_in_grid ]]) {
    // This calculates a laplace of gaussian kernel in the area surrounding the current pixel
    int y = 3 * 1200;
    int x = 3 * 1;
    float up = inVector[id - y];
    float down = inVector[id + y];
    float left = inVector[id - x];
    float right = inVector[id + x];
    float upleft = inVector[id - x - y];
    float upright = inVector[id + x - y];
    float downleft = inVector[id - x + y];
    float downright = inVector[id + x + y];
    float center = inVector[id];
    
    float laplacian = -upleft + -up + -upright + -left + (8*center) + -right + -downleft + -down + -downright;
    
    outVector[id] = sqrt(laplacian * laplacian);
}

kernel void AssembleAllInFocus(const device float *inVectorVolume [[ buffer(0) ]],
                               const device int *inVectorFocus [[ buffer(1) ]],
                               
                               constant int& length [[ buffer(2)]],
                               constant int& width [[ buffer(3)]],
                               constant int& depth [[ buffer(4)]],

                               device float *outVector [[ buffer(5) ]],
                               uint id [[ thread_position_in_grid ]]) {
    
    // This doubling needs to happen because of... byte alignment? idk man...
    // When it's not doubled, there are blank lines
    int imgIdx = inVectorFocus[id*2];
    
    int d = length * width;
    int volumeIdx = 3 * ((imgIdx * d) + id);
    float pixelR = inVectorVolume[volumeIdx+0];
    float pixelG = inVectorVolume[volumeIdx+1];
    float pixelB = inVectorVolume[volumeIdx+2];

    
    int outputIdx = id * 3;
    outVector[outputIdx+0] = pixelR;
    outVector[outputIdx+1] = pixelG;
    outVector[outputIdx+2] = pixelB;

}
