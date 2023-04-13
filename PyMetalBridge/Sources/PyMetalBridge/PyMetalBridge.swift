import Metal
import MetalPerformanceShaders
import Accelerate
import Foundation

let metallib =  "\(#file.replacingOccurrences(of: "/PyMetalBridge.swift", with: ""))/Shaders.metallib"

 @available(macOS 10.13, *)
 let device = MTLCreateSystemDefaultDevice()!,
     commandQueue = device.makeCommandQueue()!,
     defaultLibrary = try! device.makeLibrary(filepath: metallib)

@available(macOS 10.13, *)
@_cdecl("swift_focus_metric_on_gpu")
public func swift_focus_metric_on_gpu(input: UnsafePointer<Float>, output: UnsafeMutablePointer<Float>, count: Int) -> Int {
    return computeOnGPU1D(functionName: "ComputeFocusMetric", input: input, output: output, count: count)
}

@available(macOS 10.13, *)
@_cdecl("swift_focus_metric_flat_on_gpu")
public func swift_focus_metric_flat_on_gpu(input: UnsafePointer<Float>, output: UnsafeMutablePointer<Float>, count: Int) -> Int {
    return computeOnGPU1D(functionName: "ComputeFocusMetricFlat", input: input, output: output, count: count)
}

@available(macOS 10.13, *)
@_cdecl("swift_all_in_focus_on_gpu")
public func swift_all_in_focus_on_gpu(inputVolume: UnsafePointer<Float>, inputFocus: UnsafePointer<Int>, output: UnsafeMutablePointer<Float>, L: UnsafePointer<Int>, W: UnsafePointer<Int>, D: UnsafePointer<Int>) -> Int {
    return allInFocus(functionName: "AssembleAllInFocus", inputVolume: inputVolume, inputFocus: inputFocus, output: output, L: L, W: W, D: D)
}


@available(macOS 10.13, *)
func computeOnGPU1D(functionName: String,  input: UnsafePointer<Float>, output: UnsafeMutablePointer<Float>, count: Int) -> Int {
    do {
        let inputBuffer = UnsafeRawPointer(input)
        let commandBuffer = commandQueue.makeCommandBuffer()!
        let computeCommandEncoder = commandBuffer.makeComputeCommandEncoder()!

        let Function = defaultLibrary.makeFunction(name: functionName)!
        let computePipelineState = try device.makeComputePipelineState(function: Function)
        computeCommandEncoder.setComputePipelineState(computePipelineState)

        let inputByteLength = count*MemoryLayout<Float>.size

        let inVectorBuffer = device.makeBuffer(bytes: inputBuffer, length: inputByteLength, options: [])

        computeCommandEncoder.setBuffer(inVectorBuffer, offset: 0, index: 0)

        let resultRef = UnsafeMutablePointer<Float>.allocate(capacity: count)
        let outVectorBuffer = device.makeBuffer(bytes: resultRef, length: inputByteLength, options: [])

        computeCommandEncoder.setBuffer(outVectorBuffer, offset: 0, index: 1)

        let maxTotalThreadsPerThreadgroup = computePipelineState.maxTotalThreadsPerThreadgroup
        let threadExecutionWidth = computePipelineState.threadExecutionWidth
        let width  = maxTotalThreadsPerThreadgroup / threadExecutionWidth * threadExecutionWidth
        let height = 1
        let depth  = 1

        // 1D
        let threadsPerGroup = MTLSize(width:width, height: height, depth: depth)
        let numThreadgroups = MTLSize(width: (count + width - 1) / width, height: 1, depth: 1)

        computeCommandEncoder.dispatchThreadgroups(numThreadgroups, threadsPerThreadgroup: threadsPerGroup)
        computeCommandEncoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        // unsafe bitcast and assigning result pointer to output
        output.initialize(from: outVectorBuffer!.contents().assumingMemoryBound(to: Float.self), count: count)

        free(resultRef)

        return 0
    } catch {
        print("\(error)")
        return 1
    }
}

func allInFocus(functionName: String,  inputVolume: UnsafePointer<Float>, inputFocus: UnsafePointer<Int>, output: UnsafeMutablePointer<Float>, L: UnsafePointer<Int>, W: UnsafePointer<Int>, D: UnsafePointer<Int>) -> Int {
    do {
        let inputVolumeBuffer = UnsafeRawPointer(inputVolume)
        let inputFocusBuffer = UnsafeRawPointer(inputFocus)
        
        let commandBuffer = commandQueue.makeCommandBuffer()!
        let computeCommandEncoder = commandBuffer.makeComputeCommandEncoder()!

        let Function = defaultLibrary.makeFunction(name: functionName)!
        let computePipelineState = try device.makeComputePipelineState(function: Function)
        computeCommandEncoder.setComputePipelineState(computePipelineState)

        let inputVolumeByteLength = 3 * L.pointee * W.pointee * D.pointee * MemoryLayout<Float>.size
        let inputFocusByteLength = L.pointee * W.pointee * MemoryLayout<Int>.size


        let inVectorVolumeBuffer = device.makeBuffer(bytes: inputVolumeBuffer, length: inputVolumeByteLength, options: [])
        computeCommandEncoder.setBuffer(inVectorVolumeBuffer, offset: 0, index: 0)

        let inVectorFocusBuffer = device.makeBuffer(bytes: inputFocusBuffer, length: inputFocusByteLength, options: [])
        computeCommandEncoder.setBuffer(inVectorFocusBuffer, offset: 0, index: 1)

        // Send the image dimensions to the gpu
        computeCommandEncoder.setBytes(L,
                                length: MemoryLayout<Int>.stride,
                                index: 2)
        computeCommandEncoder.setBytes(W,
                                length: MemoryLayout<Int>.stride,
                                index: 3)
        computeCommandEncoder.setBytes(D,
                                length: MemoryLayout<Int>.stride,
                                index: 4)
                                 
        
        let resultRef = UnsafeMutablePointer<Float>.allocate(capacity: L.pointee * W.pointee * 3)
        let outVectorBuffer = device.makeBuffer(bytes: resultRef, length: inputFocusByteLength * 3, options: [])

        computeCommandEncoder.setBuffer(outVectorBuffer, offset: 0, index: 5)

        let maxTotalThreadsPerThreadgroup = computePipelineState.maxTotalThreadsPerThreadgroup
        let threadExecutionWidth = computePipelineState.threadExecutionWidth
        let width  = maxTotalThreadsPerThreadgroup / threadExecutionWidth * threadExecutionWidth
        let height = 1
        let depth  = 1

        // 1D
        let threadsPerGroup = MTLSize(width:width, height: height, depth: depth)
        let numThreadgroups = MTLSize(width: ((L.pointee * W.pointee) + width - 1) / width, height: 1, depth: 1)

        computeCommandEncoder.dispatchThreadgroups(numThreadgroups, threadsPerThreadgroup: threadsPerGroup)
        computeCommandEncoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        // unsafe bitcast and assigning result pointer to output
        output.initialize(from: outVectorBuffer!.contents().assumingMemoryBound(to: Float.self), count: L.pointee * W.pointee * 3)

        free(resultRef)

        return 0
    } catch {
        print("\(error)")
        return 1
    }
}
