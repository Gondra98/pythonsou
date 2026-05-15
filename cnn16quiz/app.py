import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from PIL import Image
import matplotlib.pyplot as plt

app = Flask(__name__)

# 업로드 이미지 저장 폴더 (없으면 자동 생성)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# CIFAR-100 클래스 이름 100개 (인덱스 순서 고정)
CIFAR100_CLASSES = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea',
    'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
    'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
    'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
]

# 학습된 모델 로드 (서버 시작 시 1회만 실행)
print('모델 로딩 중...')
model = tf.keras.models.load_model('cifar100_mobilenetv2.keras')
print('모델 로딩 완료')


def preprocess(img_path):
    """
    이미지 전처리 함수
    - PIL로 이미지 로드 → matplotlib으로 서버에서 시각화 확인
    - 32x32 리사이즈 → 정규화(0~1) → 배치 차원 추가
    """
    img = Image.open(img_path).convert('RGB')

    # 서버 내부에서 matplotlib으로 수신 이미지 확인 (uploads/preview.png 저장)
    plt.imshow(img)
    plt.title('수신된 이미지')
    plt.axis('off')
    plt.savefig(os.path.join(UPLOAD_FOLDER, 'preview.png'))
    plt.close()

    # CIFAR-100 입력 크기에 맞게 리사이즈 + 정규화
    img = img.resize((32, 32))
    img_array = np.array(img).astype('float32') / 255.0

    # 모델 입력: (1, 32, 32, 3) — 배치 차원 추가
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


@app.route('/')
def index():
    # index.html 렌더링 (templates/index.html)
    return render_template('index.html')


@app.route('/classify', methods=['POST'])
def classify():
    # 이미지 파일 수신 확인
    if 'image' not in request.files:
        return jsonify({'error': '이미지 파일이 없습니다'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': '파일이 선택되지 않았습니다'}), 400

    # 수신된 이미지 서버에 저장
    save_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(save_path)

    # 전처리 후 딥러닝 모델로 추론
    img_array = preprocess(save_path)
    predictions = model.predict(img_array, verbose=0)[0]  # shape: (100,)

    # 확률 상위 5개 인덱스 추출 (내림차순)
    top5_indices = predictions.argsort()[-5:][::-1]

    # 인덱스 → 클래스 이름 + 확률(%) 변환
    top5 = [[CIFAR100_CLASSES[i], float(predictions[i] * 100)] for i in top5_indices]

    # 클라이언트에 JSON 반환: top1 클래스명 + top5 리스트
    return jsonify({
        'top1': top5[0][0],
        'top5': top5
    })


if __name__ == '__main__':
    app.run(debug=True)