import coremltools
import keras.layers as KL
import numpy as np
from util import ResizeBilinear

count_lambda = 0
count_BU = 0


def keras2coreml(model, output_file="model.mlmodel"):
    from util import BatchNorm
    from tensorflow.python.keras.utils.generic_utils import CustomObjectScope
    from coremltools.proto import NeuralNetwork_pb2
    global count_BU, count_lambda

    def convert_BU(layer):
        global count_BU
        if isinstance(layer, ResizeBilinear) and layer.name:

            # params = NeuralNetwork_pb2.ReorganizeDataLayerParams()
            params = NeuralNetwork_pb2.CustomLayerParams()

            # The name of the Swift or Obj-C class that implements this layer.
            params.className = str(layer.name)

            # The desciption is shown in Xcode's mlmodel viewer.
            params.description = "This a BilinearUpsampling layer transformed by CoreML"

            count_BU += 1
            return params

        else:
            return None

    def convert_lambda(layer):
        global count_lambda
        if isinstance(layer, KL.Lambda) and layer.name:
            print('-------LAMBDA--------')

            params = NeuralNetwork_pb2.CustomLayerParams()

            # The name of the Swift or Obj-C class that implements this layer.
            params.className = str(layer.name)
            # The desciption is shown in Xcode's mlmodel viewer.
            params.description = "This a Lambda layer transformed by CoreML"

            count_lambda += 1
            return params
        elif isinstance(layer, KL.Lambda) and not layer.name:
            print('-------LAMBDA--------')

            params = NeuralNetwork_pb2.CustomLayerParams()

            # The name of the Swift or Obj-C class that implements this layer.
            params.className = 'Lambda'
            # The desciption is shown in Xcode's mlmodel viewer.
            params.description = "This a Lambda layer transformed by CoreML"


            count_lambda += 1
            return params
        else:
            return None

    with CustomObjectScope({#'BilinearUpsampling': BilinearUpsampling
         }):
        coreml_model = coremltools.converters.keras.convert(
            model.keras_model,
            input_names='image',
            image_input_names='image',
            output_names='matting',
            add_custom_layers=True,
            custom_conversion_functions={"Lambda": convert_lambda,
                                         'ResizeBilinear': convert_BU
                                         },

        )

        print('\n\n\n')
        for i, layer in enumerate(coreml_model._spec.neuralNetwork.layers):
            if layer.HasField("custom"):
                print("Layer %d = %s --> custom layer = %s" % (i, layer.name, layer.custom.className))
            else:
                print("Layer %d = %s" % (i, layer.name))

        # setup the attribution meta-data for the model
        coreml_model.author = 'yunke zhang'
        coreml_model.short_description = 'A Late Fusion CNN for Digital Matting.'
        coreml_model.input_description['image'] = 'An input image in RGB order'
        coreml_model.output_description['matting'] = 'The segmentation map as the matting output'


        coreml_model.save(output_file)
        print(f' number of Lambda : {count_lambda}\n number of BU : {count_BU}')
