## 使用TensorFlow Lite Model Maker生成图像分类模型
1. 首先安装程序运行必备的一些库
```
!pip install tflite-model-maker
```
```
Requirement already satisfied: tflite-model-maker in /opt/conda/envs/tf/lib/python3.8/site-packages (0.4.2)
Requirement already satisfied: Cython>=0.29.13 in /opt/conda/envs/tf/lib/python3.8/site-packages (from tflite-model-maker) (0.29.35)
Requirement already satisfied: tf-models-official==2.3.0 in /opt/conda/envs/tf/lib/python3.8/site-packages (from tflite-model-maker) (2.3.0)
Requirement already satisfied: tensorflow-model-optimization>=0.5 in /opt/conda/envs/tf/lib/python3.8/site-packages (from tflite-model-maker) (0.7.5)
Requirement already satisfied: tensorflow-hub<0.13,>=0.7.0 in /opt/conda/envs/tf/lib/python3.8/site-packages (from tflite-model-maker) (0.12.0)
Requirement already satisfied: pillow>=7.0.0 in /opt/conda/envs/tf/lib/python3.8/site-packages (from tflite-model-maker) (9.5.0)
Requirement already satisfied: tensorflowjs<3.19.0,>=2.4.0 in /opt/conda/envs/tf/lib/python3.8/site-packages (from tflite-model-maker) (3.18.0)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/conda/envs/tf/lib/python3.8/site-packages (from tflite-model-maker) (1.25.11)
Requirement already satisfied: sentencepiece>=0.1.91 in /opt/conda/envs/tf/lib/python3.8/site-packages (from tflite-model-maker) (0.1.99)
Requirement already satisfied: tensorflow-datasets>=2.1.0 in /opt/conda/envs/tf/lib/python3.8/site-packages (from tflite-model-maker) (4.9.0)
Requirement already satisfied: scann==1.2.6 in /opt/conda/envs/tf/lib/python3.8/site-packages (from tflite-model-maker) (1.2.6)
Requirement already satisfied: PyYAML>=5.1 in /opt/conda/envs/tf/lib/python3.8/site-packages (from tflite-model-maker) (6.0)
Requirement already satisfied: librosa==0.8.1 in /opt/conda/envs/tf/lib/python3.8/site-packages (from tflite-model-maker) (0.8.1)
Requirement already satisfied: matplotlib<3.5.0,>=3.0.3 in /opt/conda/envs/tf/lib/python3.8/site-packages (from tflite-model-maker) (3.4.3)
Requirement already satisfied: tflite-support>=0.4.2 in /opt/conda/envs/tf/lib/python3.8/site-packages (from tflite-model-maker) (0.4.3)
Requirement already satisfied: absl-py>=0.10.0 in /opt/conda/envs/tf/lib/python3.8/site-packages (from tflite-model-maker) (1.4.0)
Requirement already satisfied: tensorflow>=2.6.0 in /opt/conda/envs/tf/lib/python3.8/site-packages (from tflite-model-maker) (2.8.4)
Requirement already satisfied: lxml>=4.6.1 in /opt/conda/envs/tf/lib/python3.8/site-packages (from tflite-model-maker) (4.9.2)
Requirement already satisfied: numba==0.53 in /opt/conda/envs/tf/lib/python3.8/site-packages (from tflite-model-maker) (0.53.0)
Requirement already satisfied: neural-structured-learning>=1.3.1 in /opt/conda/envs/tf/lib/python3.8/site-packages (from tflite-model-maker) (1.4.0)
Requirement already satisfied: tensorflow-addons>=0.11.2 in /opt/conda/envs/tf/lib/python3.8/site-packages (from tflite-model-maker) (0.20.0)
Requirement already satisfied: flatbuffers>=2.0 in /opt/conda/envs/tf/lib/python3.8/site-packages (from tflite-model-maker) (23.5.26)
Requirement already satisfied: fire>=0.3.1 in /opt/conda/envs/tf/lib/python3.8/site-packages (from tflite-model-maker) (0.5.0)
Requirement already satisfied: numpy>=1.17.3 in /opt/conda/envs/tf/lib/python3.8/site-packages (from tflite-model-maker) (1.23.0)
Requirement already satisfied: six>=1.12.0 in /opt/conda/envs/tf/lib/python3.8/site-packages (from tflite-model-maker) (1.16.0)
...
    Uninstalling packaging-23.1:
      Successfully uninstalled packaging-23.1
Successfully installed packaging-20.9
Note: you may need to restart the kernel to use updated packages.
```
```
!pip install conda-repo-cli==1.0.4
```
```
Requirement already satisfied: conda-repo-cli==1.0.4 in d:\anaconda\envs\tf\lib\site-packages (1.0.4)
Requirement already satisfied: PyYAML>=3.12 in d:\anaconda\envs\tf\lib\site-packages (from conda-repo-cli==1.0.4) (6.0)
Requirement already satisfied: requests>=2.9.1 in d:\anaconda\envs\tf\lib\site-packages (from conda-repo-cli==1.0.4) (2.31.0)
Requirement already satisfied: pytz in d:\anaconda\envs\tf\lib\site-packages (from conda-repo-cli==1.0.4) (2023.3)
Requirement already satisfied: six in d:\anaconda\envs\tf\lib\site-packages (from conda-repo-cli==1.0.4) (1.16.0)
Requirement already satisfied: pathlib in d:\anaconda\envs\tf\lib\site-packages (from conda-repo-cli==1.0.4) (1.0.1)
Requirement already satisfied: nbformat>=4.4.0 in d:\anaconda\envs\tf\lib\site-packages (from conda-repo-cli==1.0.4) (5.7.0)
Requirement already satisfied: setuptools in d:\anaconda\envs\tf\lib\site-packages (from conda-repo-cli==1.0.4) (66.0.0)
Requirement already satisfied: clyent>=1.2.0 in d:\anaconda\envs\tf\lib\site-packages (from conda-repo-cli==1.0.4) (1.2.2)
Requirement already satisfied: python-dateutil>=2.6.1 in d:\anaconda\envs\tf\lib\site-packages (from conda-repo-cli==1.0.4) (2.8.2)
Requirement already satisfied: jupyter-core in d:\anaconda\envs\tf\lib\site-packages (from nbformat>=4.4.0->conda-repo-cli==1.0.4) (5.3.0)
Requirement already satisfied: fastjsonschema in d:\anaconda\envs\tf\lib\site-packages (from nbformat>=4.4.0->conda-repo-cli==1.0.4) (2.16.2)
Requirement already satisfied: traitlets>=5.1 in d:\anaconda\envs\tf\lib\site-packages (from nbformat>=4.4.0->conda-repo-cli==1.0.4) (5.9.0)
Requirement already satisfied: jsonschema>=2.6 in d:\anaconda\envs\tf\lib\site-packages (from nbformat>=4.4.0->conda-repo-cli==1.0.4) (4.17.3)
Requirement already satisfied: urllib3<3,>=1.21.1 in d:\anaconda\envs\tf\lib\site-packages (from requests>=2.9.1->conda-repo-cli==1.0.4) (1.26.15)
Requirement already satisfied: idna<4,>=2.5 in d:\anaconda\envs\tf\lib\site-packages (from requests>=2.9.1->conda-repo-cli==1.0.4) (3.4)
Requirement already satisfied: certifi>=2017.4.17 in d:\anaconda\envs\tf\lib\site-packages (from requests>=2.9.1->conda-repo-cli==1.0.4) (2023.5.7)
Requirement already satisfied: charset-normalizer<4,>=2 in c:\users\17764\appdata\roaming\python\python38\site-packages (from requests>=2.9.1->conda-repo-cli==1.0.4) (3.1.0)
Requirement already satisfied: pkgutil-resolve-name>=1.3.10 in d:\anaconda\envs\tf\lib\site-packages (from jsonschema>=2.6->nbformat>=4.4.0->conda-repo-cli==1.0.4) (1.3.10)
Requirement already satisfied: pyrsistent!=0.17.0,!=0.17.1,!=0.17.2,>=0.14.0 in d:\anaconda\envs\tf\lib\site-packages (from jsonschema>=2.6->nbformat>=4.4.0->conda-repo-cli==1.0.4) (0.18.0)
Requirement already satisfied: importlib-resources>=1.4.0 in d:\anaconda\envs\tf\lib\site-packages (from jsonschema>=2.6->nbformat>=4.4.0->conda-repo-cli==1.0.4) (5.12.0)
Requirement already satisfied: attrs>=17.4.0 in d:\anaconda\envs\tf\lib\site-packages (from jsonschema>=2.6->nbformat>=4.4.0->conda-repo-cli==1.0.4) (23.1.0)
Requirement already satisfied: platformdirs>=2.5 in d:\anaconda\envs\tf\lib\site-packages (from jupyter-core->nbformat>=4.4.0->conda-repo-cli==1.0.4) (3.5.1)
Requirement already satisfied: pywin32>=300 in d:\anaconda\envs\tf\lib\site-packages (from jupyter-core->nbformat>=4.4.0->conda-repo-cli==1.0.4) (305.1)
Requirement already satisfied: zipp>=3.1.0 in d:\anaconda\envs\tf\lib\site-packages (from importlib-resources>=1.4.0->jsonschema>=2.6->nbformat>=4.4.0->conda-repo-cli==1.0.4) (3.15.0)
```
```
!pip install anaconda-project==0.10.1
```
```
Requirement already satisfied: anaconda-project==0.10.1 in d:\anaconda\envs\tf\lib\site-packages (0.10.1)
Requirement already satisfied: jinja2 in d:\anaconda\envs\tf\lib\site-packages (from anaconda-project==0.10.1) (3.1.2)
Requirement already satisfied: requests in d:\anaconda\envs\tf\lib\site-packages (from anaconda-project==0.10.1) (2.31.0)
Requirement already satisfied: tornado>=4.2 in d:\anaconda\envs\tf\lib\site-packages (from anaconda-project==0.10.1) (6.2)
Requirement already satisfied: conda-pack in d:\anaconda\envs\tf\lib\site-packages (from anaconda-project==0.10.1) (0.6.0)
Requirement already satisfied: ruamel-yaml in d:\anaconda\envs\tf\lib\site-packages (from anaconda-project==0.10.1) (0.17.28)
Requirement already satisfied: anaconda-client in d:\anaconda\envs\tf\lib\site-packages (from anaconda-project==0.10.1) (1.11.2)
Requirement already satisfied: clyent>=1.2.0 in d:\anaconda\envs\tf\lib\site-packages (from anaconda-client->anaconda-project==0.10.1) (1.2.2)
Requirement already satisfied: conda-package-handling>=1.7.3 in d:\anaconda\envs\tf\lib\site-packages (from anaconda-client->anaconda-project==0.10.1) (2.1.0)
Requirement already satisfied: defusedxml>=0.7.1 in d:\anaconda\envs\tf\lib\site-packages (from anaconda-client->anaconda-project==0.10.1) (0.7.1)
Requirement already satisfied: nbformat>=4.4.0 in d:\anaconda\envs\tf\lib\site-packages (from anaconda-client->anaconda-project==0.10.1) (5.7.0)
Requirement already satisfied: python-dateutil>=2.6.1 in d:\anaconda\envs\tf\lib\site-packages (from anaconda-client->anaconda-project==0.10.1) (2.8.2)
Requirement already satisfied: pytz>=2021.3 in d:\anaconda\envs\tf\lib\site-packages (from anaconda-client->anaconda-project==0.10.1) (2023.3)
Requirement already satisfied: pyyaml>=3.12 in d:\anaconda\envs\tf\lib\site-packages (from anaconda-client->anaconda-project==0.10.1) (6.0)
Requirement already satisfied: requests-toolbelt>=0.9.1 in d:\anaconda\envs\tf\lib\site-packages (from anaconda-client->anaconda-project==0.10.1) (0.9.1)
Requirement already satisfied: setuptools>=58.0.4 in d:\anaconda\envs\tf\lib\site-packages (from anaconda-client->anaconda-project==0.10.1) (66.0.0)
Requirement already satisfied: six>=1.15.0 in d:\anaconda\envs\tf\lib\site-packages (from anaconda-client->anaconda-project==0.10.1) (1.16.0)
Requirement already satisfied: tqdm>=4.56.0 in d:\anaconda\envs\tf\lib\site-packages (from anaconda-client->anaconda-project==0.10.1) (4.65.0)
Requirement already satisfied: urllib3>=1.26.4 in d:\anaconda\envs\tf\lib\site-packages (from anaconda-client->anaconda-project==0.10.1) (1.26.15)
Requirement already satisfied: idna<4,>=2.5 in d:\anaconda\envs\tf\lib\site-packages (from requests->anaconda-project==0.10.1) (3.4)
Requirement already satisfied: certifi>=2017.4.17 in d:\anaconda\envs\tf\lib\site-packages (from requests->anaconda-project==0.10.1) (2023.5.7)
Requirement already satisfied: charset-normalizer<4,>=2 in c:\users\17764\appdata\roaming\python\python38\site-packages (from requests->anaconda-project==0.10.1) (3.1.0)
Requirement already satisfied: MarkupSafe>=2.0 in d:\anaconda\envs\tf\lib\site-packages (from jinja2->anaconda-project==0.10.1) (2.1.2)
Requirement already satisfied: ruamel.yaml.clib>=0.2.7 in d:\anaconda\envs\tf\lib\site-packages (from ruamel-yaml->anaconda-project==0.10.1) (0.2.7)
...
Requirement already satisfied: attrs>=17.4.0 in d:\anaconda\envs\tf\lib\site-packages (from jsonschema>=2.6->nbformat>=4.4.0->anaconda-client->anaconda-project==0.10.1) (23.1.0)
Requirement already satisfied: platformdirs>=2.5 in d:\anaconda\envs\tf\lib\site-packages (from jupyter-core->nbformat>=4.4.0->anaconda-client->anaconda-project==0.10.1) (3.5.1)
Requirement already satisfied: pywin32>=300 in d:\anaconda\envs\tf\lib\site-packages (from jupyter-core->nbformat>=4.4.0->anaconda-client->anaconda-project==0.10.1) (305.1)
Requirement already satisfied: zipp>=3.1.0 in d:\anaconda\envs\tf\lib\site-packages (from importlib-resources>=1.4.0->jsonschema>=2.6->nbformat>=4.4.0->anaconda-client->anaconda-project==0.10.1) (3.15.0)
```
2. 接下来，导入相关的库，这里注意numpy.object已经在numpy1.20版本被弃用，直接使用object代替
```
import os

import numpy as np
object = np.object
import tensorflow as tf
assert tf.__version__.startswith('2')

from tflite_model_maker import model_spec
from tflite_model_maker import image_classifier
from tflite_model_maker.config import ExportFormat
from tflite_model_maker.config import QuantizationConfig
from tflite_model_maker.image_classifier import DataLoader

import matplotlib.pyplot as plt
```
3. 获取数据
```
image_path = tf.keras.utils.get_file(
      'flower_photos.tgz',
      'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
      extract=True)
image_path = os.path.join(os.path.dirname(image_path), 'flower_photos')

```
4. 运行示例
   1. 第一步：加载数据集，并将数据集分为训练数据和测试数据。
    ``` 
    data = DataLoader.from_folder(image_path)
    train_data, test_data = data.split(0.9)
    ```
    ```
    INFO:tensorflow:Load image with size: 3670, num_label: 5, labels: daisy, dandelion, roses, sunflowers, tulips.
    ```
    2. 第二步：训练Tensorflow模型
      ```
      model = image_classifier.create(train_data)
      ```
      ```
        INFO:tensorflow:Retraining the models...
        Model: "sequential"
        _________________________________________________________________
        Layer (type)                Output Shape              Param #   
        =================================================================
        hub_keras_layer_v1v2 (HubKe  (None, 1280)             3413024   
        rasLayerV1V2)                                                   
                                                                        
        dropout (Dropout)           (None, 1280)              0         
                                                                        
        dense (Dense)               (None, 5)                 6405      
                                                                        
        =================================================================
        Total params: 3,419,429
        Trainable params: 6,405
        Non-trainable params: 3,413,024
        _________________________________________________________________
        None
        Epoch 1/5
        2023-05-31 02:18:27.394135: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 51380224 exceeds 10% of free system memory.
        2023-05-31 02:18:27.719422: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 51380224 exceeds 10% of free system memory.
        2023-05-31 02:18:27.760035: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 51380224 exceeds 10% of free system memory.
        2023-05-31 02:18:27.791106: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 25690112 exceeds 10% of free system memory.
        2023-05-31 02:18:27.803388: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 154140672 exceeds 10% of free system memory.
        103/103 [==============================] - 56s 523ms/step - loss: 0.8544 - accuracy: 0.7788
        Epoch 2/5
        103/103 [==============================] - 53s 509ms/step - loss: 0.6597 - accuracy: 0.8956
        Epoch 3/5
        103/103 [==============================] - 51s 498ms/step - loss: 0.6207 - accuracy: 0.9123
        Epoch 4/5
        103/103 [==============================] - 52s 499ms/step - loss: 0.6040 - accuracy: 0.9281
        Epoch 5/5
        103/103 [==============================] - 51s 496ms/step - loss: 0.5887 - accuracy: 0.9345


        103/103 [==============================] - 76s 719ms/step - loss: 0.8647 - accuracy: 0.7782
        Epoch 2/5
        103/103 [==============================] - 97s 943ms/step - loss: 0.6525 - accuracy: 0.8935
        Epoch 3/5
        103/103 [==============================] - 92s 896ms/step - loss: 0.6223 - accuracy: 0.9099
        Epoch 4/5
        103/103 [==============================] - 95s 921ms/step - loss: 0.6021 - accuracy: 0.9226
        Epoch 5/5
        103/103 [==============================] - 100s 970ms/step - loss: 0.5903 - accuracy: 0.9336

      ```
    3. 第三步：评估模型
      ```
      loss, accuracy = model.evaluate(test_data)
      ```
      ```
      12/12 [==============================] - 11s 485ms/step - loss: 0.6107 - accuracy: 0.9155
      ```
    4. 第四步，导出Tensorflow Lite模型
      ```
      model.export(export_dir='.')
      ```
      ```
       2023-05-31 02:28:55.579868: W tensorflow/python/util/util.cc:368] Sets are not currently considered sequences, but this may change in the future, so consider avoiding using them.
      INFO:tensorflow:Assets written to: /tmp/tmpxb2d4_22/assets
      INFO:tensorflow:Assets written to: /tmp/tmpxb2d4_22/assets
      2023-05-31 02:29:00.350357: I tensorflow/core/grappler/devices.cc:66] Number of eligible GPUs (core count >= 8, compute capability >= 0.0): 0
      2023-05-31 02:29:00.350578: I tensorflow/core/grappler/clusters/single_machine.cc:358] Starting new session
      2023-05-31 02:29:00.411724: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:1164] Optimization results for grappler item: graph_to_optimize
        function_optimizer: Graph size after: 913 nodes (656), 923 edges (664), time = 24.347ms.
        function_optimizer: function_optimizer did nothing. time = 0.013ms.

      /opt/conda/envs/tf/lib/python3.8/site-packages/tensorflow/lite/python/convert.py:746: UserWarning: Statistics for quantized inputs were expected, but not specified; continuing anyway.
        warnings.warn("Statistics for quantized inputs were expected, but not "
      2023-05-31 02:29:01.222461: W tensorflow/compiler/mlir/lite/python/tf_tfl_flatbuffer_helpers.cc:357] Ignored output_format.
      2023-05-31 02:29:01.222508: W tensorflow/compiler/mlir/lite/python/tf_tfl_flatbuffer_helpers.cc:360] Ignored drop_control_dependency.
      INFO:tensorflow:Label file is inside the TFLite model with metadata.
      fully_quantize: 0, inference_type: 6, input_inference_type: 3, output_inference_type: 3
      INFO:tensorflow:Label file is inside the TFLite model with metadata.
      INFO:tensorflow:Saving labels in /tmp/tmp9w7lih1u/labels.txt
      INFO:tensorflow:Saving labels in /tmp/tmp9w7lih1u/labels.txt
      INFO:tensorflow:TensorFlow Lite model exported successfully: ./model.tflite
      INFO:tensorflow:TensorFlow Lite model exported successfully: ./model.tflite

      ```
