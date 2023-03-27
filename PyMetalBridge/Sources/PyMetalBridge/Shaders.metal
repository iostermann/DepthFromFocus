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

kernel void sigmoid(const device float *inVector [[ buffer(0) ]],
                    device float *outVector [[ buffer(1) ]],
                    uint id [[ thread_position_in_grid ]]) {
    // This calculates sigmoid for _one_ position (=id) in a vector per call on the GPU
    outVector[id] = 1.0 / (1.0 + exp(-inVector[id]));
}

inline int factorial(int n) {
  int product = 1;
  for(int i = 1; i < n; ++i ) {
      product *= i;
  }
  return product;
}

kernel void maclaurin_cos(const device float *inVector [[ buffer(0) ]],
                    device float *outVector [[ buffer(1) ]],
                    uint id [[ thread_position_in_grid ]]) {

    float approximate = 0;
    for(int i = 0; i < 10; i++) {
        float x = inVector[id];
        float coef = pow(-1.0f, i);
        int num = pow(x, 2.0f*i);
        int denom = factorial(2.0f*i);
        approximate += coef * (num/denom);
        i++;
    }
    outVector[id] = approximate;
}

inline float f(const float x) {
    float approximate = 0;
    for(int coeff = 1; coeff < 10000; coeff+=2) {
        approximate += (1.0f/coeff)*sin(coeff*x);
    }
    return approximate;
}

constant float delta = 1e-4;

kernel void differential(const device float *inVector [[ buffer(0) ]],
                    device float *outVector [[ buffer(1) ]],
                    uint id [[ thread_position_in_grid ]]) {

    float x = inVector[id];
    outVector[id] = (f(x+delta) - f(x-delta)) / 2.0f*delta;
}
