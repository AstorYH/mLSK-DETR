import warnings
import wandb
wandb.init(mode="offline")

warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    # model = RTDETR('ultralytics/cfg/models/rt-detr/myrtdetr.yaml')
    # model = RTDETR(r'E:\objectdetection\RTDETR-m\ultralytics\cfg\models\rt-detr\rtdetr-r50.yaml')
    model = RTDETR('ultralytics/cfg/models/rt-detr/rtdetr-lsknet.yaml')
    # model = RTDETR(r'E:\objectdetection\RTDETR-m\ultralytics\cfg\models\rt-detr\rtdetr-lsknet.yaml')
    # model = RTDETR('ultralytics/cfg/models/yolo-detr/yolov8-detr.yaml')
    # model.load('') # loading pretrain weights
    model.train(data='dataset/MOB.yaml',
                cache=False,
                imgsz=640,
                epochs=100,
                batch=16,
                workers=14,
                device='0',
                # resume=r'E:\objectdetection\RTDETR-m\runs\train\testmyrtdetrmobdrone\weights\last.pt', # last.pt path
                project='runs/train/MOB',
                # name='testrtdetrwsodd',
                name='testlsknetmobdrone',
                )
