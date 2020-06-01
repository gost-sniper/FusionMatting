import Foundation
import CoreML
import Accelerate




@objc(deBG_side_0_out) class deBG_side_0_out: NSObject, MLCustomLayer {
      required init(parameters: [String : Any]) throws {
      print(#function, parameters)
      super.init()
  }
  
  func setWeightData(_ weights: [Data]) throws {
      //print(#function, weights)
  }
  
  func outputShapes(forInputShapes inputShapes: [[NSNumber]]) throws -> [[NSNumber]] {
      print(#function, inputShapes)
              
    let outputShape = [[1, 1, inputShapes[0][2], 640, 960]]
      
    print("Upsample output shape", outputShape)

    return outputShape
  }
  
  func bilinear(src_mlma: MLMultiArray, dst_mlma: MLMultiArray, index: Int, src_w: Int, src_h: Int, dst_w: Int, dst_h: Int) {
      var src = UnsafeMutablePointer<Float>(OpaquePointer(src_mlma.dataPointer))
      src = src.advanced(by: index*src_w*src_h)
      var dst = UnsafeMutablePointer<Float>(OpaquePointer(dst_mlma.dataPointer))
      dst = dst.advanced(by: index*dst_w*dst_h)
      
      let h_ratio = (Float)(dst_h) / (Float)(src_h)
      let w_ratio = (Float)(dst_w) / (Float)(src_w)
      
      for y in 0..<dst_h {
          for x in 0..<dst_w {
              var px = (Int)((Float)(x) / w_ratio)
              var py = (Int)((Float)(y) / h_ratio)
              
              if px >= src_w - 1 {
                  px = src_w - 2
              }
              
              if py >= src_h - 1 {
                  py = src_h - 2
              }
              
              let fx1 = (Float)(x) / w_ratio - (Float)(px)
              let fx2 = 1.0 - fx1
              let fy1 = (Float)(y) / h_ratio - (Float)(py)
              let fy2 = 1.0 - fy1
              
              let w1 = fx2 * fy2
              let w2 = fx1 * fy2
              let w3 = fx2 * fy1
              let w4 = fx1 * fy1
              
              let p1 = src[py*src_w + px]
              let p2 = src[py*src_w + px + 1]
              let p3 = src[(py + 1)*src_w + px]
              let p4 = src[(py + 1)*src_w + px + 1]
              dst[y*dst_w + x] = w1*p1 + w2*p2 + w3*p3 + w4*p4
          }
      }
  }
  
  func evaluate(inputs: [MLMultiArray], outputs: [MLMultiArray]) throws {
      //print("Upsample", inputs[0].shape, outputs[0].shape)
      
      for i in 0..<inputs[0].shape[2].intValue {
          bilinear(src_mlma: inputs[0], dst_mlma: outputs[0], index: i, src_w: inputs[0].shape[4].intValue, src_h: inputs[0].shape[3].intValue, dst_w: outputs[0].shape[4].intValue, dst_h: outputs[0].shape[3].intValue)
      }
  }
}

@objc(deFG_side_0_out) class deFG_side_0_out: NSObject, MLCustomLayer {
  required init(parameters: [String : Any]) throws {
      print(#function, parameters)
      super.init()
  }
  
  func setWeightData(_ weights: [Data]) throws {
      //print(#function, weights)
  }
  
  func outputShapes(forInputShapes inputShapes: [[NSNumber]]) throws -> [[NSNumber]] {
      print(#function, inputShapes)
              
      let outputShape = [[1, 1, inputShapes[0][2], 640, 960]]
      
      print("Upsample output shape", outputShape)

      return outputShape
  }
  
  func bilinear(src_mlma: MLMultiArray, dst_mlma: MLMultiArray, index: Int, src_w: Int, src_h: Int, dst_w: Int, dst_h: Int) {
      var src = UnsafeMutablePointer<Float>(OpaquePointer(src_mlma.dataPointer))
      src = src.advanced(by: index*src_w*src_h)
      var dst = UnsafeMutablePointer<Float>(OpaquePointer(dst_mlma.dataPointer))
      dst = dst.advanced(by: index*dst_w*dst_h)
      
      let h_ratio = (Float)(dst_h) / (Float)(src_h)
      let w_ratio = (Float)(dst_w) / (Float)(src_w)
      
      for y in 0..<dst_h {
          for x in 0..<dst_w {
              var px = (Int)((Float)(x) / w_ratio)
              var py = (Int)((Float)(y) / h_ratio)
              
              if px >= src_w - 1 {
                  px = src_w - 2
              }
              
              if py >= src_h - 1 {
                  py = src_h - 2
              }
              
              let fx1 = (Float)(x) / w_ratio - (Float)(px)
              let fx2 = 1.0 - fx1
              let fy1 = (Float)(y) / h_ratio - (Float)(py)
              let fy2 = 1.0 - fy1
              
              let w1 = fx2 * fy2
              let w2 = fx1 * fy2
              let w3 = fx2 * fy1
              let w4 = fx1 * fy1
              
              let p1 = src[py*src_w + px]
              let p2 = src[py*src_w + px + 1]
              let p3 = src[(py + 1)*src_w + px]
              let p4 = src[(py + 1)*src_w + px + 1]
              dst[y*dst_w + x] = w1*p1 + w2*p2 + w3*p3 + w4*p4
          }
      }
  }
  
  func evaluate(inputs: [MLMultiArray], outputs: [MLMultiArray]) throws {
      //print("Upsample", inputs[0].shape, outputs[0].shape)
      
      for i in 0..<inputs[0].shape[2].intValue {
          bilinear(src_mlma: inputs[0], dst_mlma: outputs[0], index: i, src_w: inputs[0].shape[4].intValue, src_h: inputs[0].shape[3].intValue, dst_w: outputs[0].shape[4].intValue, dst_h: outputs[0].shape[3].intValue)
      }
  }
}


@objc(fusion_reverse_lambda_bg) class fusion_reverse_lambda_bg: NSObject, MLCustomLayer {


  required init(parameters: [String : Any]) throws {
    print(NSStringFromClass(type(of: self)), #function, parameters)
    super.init()
  }


  func outputShapes(forInputShapes inputShapes: [[NSNumber]]) throws -> [[NSNumber]] {
    print(NSStringFromClass(type(of: self)), #function, inputShapes)
    return inputShapes
  }
  
  func setWeightData(_ weights: [Data]) throws {
  	print(NSStringFromClass(type(of: self)), #function, weights)
        
  }

  func evaluate(inputs: [MLMultiArray], outputs: [MLMultiArray]) throws {
    for i in 0..<inputs.count {
      let input = inputs[i]
      let output = outputs[i]

      assert(input.dataType == .float32)
      assert(output.dataType == .float32)
      assert(input.shape == output.shape)

      for j in 0..<input.count {
        let x = input[j].floatValue
        let y = 1 - x
        output[j] = NSNumber(value: y)
      }
    }  
  }
}

@objc(fusion_reverse_lambda_blendingweight) class fusion_reverse_lambda_blendingweight: NSObject, MLCustomLayer {
    func setWeightData(_ weights: [Data]) throws {
        print()
    }

  required init(parameters: [String : Any]) throws {
    print(#function, parameters)
    super.init()
  }


  func outputShapes(forInputShapes inputShapes: [[NSNumber]]) throws 
       -> [[NSNumber]] {
    print(#function, inputShapes)
    return inputShapes
  }

  func evaluate(inputs: [MLMultiArray], outputs: [MLMultiArray]) throws {
    for i in 0..<inputs.count {
      let input = inputs[i]
      let output = outputs[i]

      assert(input.dataType == .float32)
      assert(output.dataType == .float32)
      assert(input.shape == output.shape)

      for j in 0..<input.count {
        let x = input[j].floatValue
        let y = 1 - x
        output[j] = NSNumber(value: y)
      }
    }  
  }
}
