import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    model = RTDETR('runs/train/RT-DETR_DA_LSK/weights/best.pt') # select your model.pt path
    model.predict(source='dataset/527测试图像',
                  project='runs/detect',
                  name='5.27测试结果',
                  save=True,
                #   visualize=True # visualize model features maps
                  )