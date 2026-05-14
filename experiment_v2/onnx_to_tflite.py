"""ONNX -> TFLite conversion using onnxruntime + numpy manual approach"""
import numpy as np
import os

CKPT = os.path.join(os.path.dirname(__file__), "checkpoints")
ONNX_PATH = os.path.join(CKPT, "model.onnx")
TFLITE_PATH = os.path.join(CKPT, "model.tflite")

# Method 1: Try onnx2tf
try:
    import onnx2tf
    onnx2tf.convert(
        input_onnx_file_path=ONNX_PATH,
        output_folder_path=os.path.join(CKPT, "tf_model"),
        non_verbose=True,
    )
    # Find the tflite file
    import glob
    tflites = glob.glob(os.path.join(CKPT, "tf_model", "*.tflite"))
    if tflites:
        import shutil
        shutil.copy(tflites[0], TFLITE_PATH)
        print(f"TFLite kaydedildi: {TFLITE_PATH}")
        print(f"Boyut: {os.path.getsize(TFLITE_PATH)/1024:.0f} KB")
        exit(0)
except Exception as e:
    print(f"onnx2tf basarisiz: {e}")

# Method 2: Try tensorflow directly
try:
    import tensorflow as tf
    import onnx
    from onnx_tf.backend import prepare

    model = onnx.load(ONNX_PATH)
    tf_rep = prepare(model)
    tf_path = os.path.join(CKPT, "model_tf")
    tf_rep.export_graph(tf_path)

    converter = tf.lite.TFLiteConverter.from_saved_model(tf_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    with open(TFLITE_PATH, 'wb') as f:
        f.write(tflite_model)
    print(f"TFLite kaydedildi: {TFLITE_PATH}")
    print(f"Boyut: {len(tflite_model)/1024:.0f} KB")
    exit(0)
except Exception as e:
    print(f"onnx-tf basarisiz: {e}")

# Method 3: Use onnxruntime directly and skip TFLite
# Instead, we'll convert to a simpler format - just save weights as numpy
print("\nTFLite donusumu basarisiz. ONNX Runtime ile calisacagiz.")
print(f"ONNX model mevcut: {ONNX_PATH}")
print(f"Boyut: {os.path.getsize(ONNX_PATH)/1024:.0f} KB")
print("\nFlutter'da onnxruntime kullanilabilir.")
