import tensorflow as tf
import numpy as np
from PIL import Image
import os
import pandas as pd

# oneDNN 옵션 비활성화
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# SavedModel 로드
model = tf.saved_model.load('saved_model/')
infer = model.signatures['serving_default']

# 예제: input name 확인 (debug용)
print(infer.structured_input_signature)

# 테스트용
img_path = 'dataset_test/testimgs/1.png'
img = Image.open(img_path).convert("L")
img = img.resize((28, 28))
im2arr = np.array(img).reshape(1, 28, 28, 1).astype('float32') / 255.0

# signature에 맞게 텐서로 입력
input_tensor = tf.convert_to_tensor(im2arr)

# 예측
pred = infer(input_tensor)

# 출력 텐서에서 값 가져오기
for key in pred:
    y_pred = pred[key].numpy()

pred_label = np.argmax(y_pred)
print("predicted label:", pred_label)
