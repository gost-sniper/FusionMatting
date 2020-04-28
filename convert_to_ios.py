import coremltools


def keras2coreml(model, output_file="model.mlmodel"):
    from util import BatchNorm
    from tensorflow.python.keras.utils.generic_utils import CustomObjectScope
    from coremltools.proto import NeuralNetwork_pb2

    def convert_lambda(layer):
        if isinstance(layer, KL.Lambda) and layer.name:

            params = NeuralNetwork_pb2.CustomLayerParams()

            # The name of the Swift or Obj-C class that implements this layer.
            params.className = str(layer.name)

            # The desciption is shown in Xcode's mlmodel viewer.
            params.description = "This a Lambda layer transformed by CoreML"

            return params
        else:
            return None

    with CustomObjectScope({'batchNorm': BatchNorm}):
        coreml_model = coremltools.converters.keras.convert(
            model.keras_model,
            input_names=['image'], output_names=['image'],
            image_input_names='image',
            # add_custom_layers=True,
            custom_conversion_functions={"Lambda": convert_lambda}
        )

        coreml_model.save(output_file)