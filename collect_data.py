import cv2
import mediapipe as mp
import numpy as np
import joblib
import torch
import pathlib
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# データ収集用のリスト
X = []
y = []

vpath = 0 # 指定のパスを入れてくだしあ

cap = cv2.VideoCapture(vpath)

with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5) as hands:

    model_path = "chopstick.pt"

    device = select_device("")
    stick_model = DetectMultiBackend(model_path, device=device)
    model_names = stick_model.names if hasattr(stick_model, "names") else stick_model.module.names

    while cap.isOpened():
        ret, image = cap.read()
        if not ret:
            break

        # BGR画像をRGBに変換
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # 画像のパディング
        origin_height, origin_width = image.shape[:2]
        size = 640
            
        # 黒色にパディングをするための方法をchatGPT4にたずねました
        if origin_width >= origin_height:
            re_image = cv2.copyMakeBorder(image, 0, (origin_width - origin_height), 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        else:
            re_image = cv2.copyMakeBorder(image, 0, 0, 0, (origin_height - origin_width), cv2.BORDER_CONSTANT, value=[0, 0, 0])

        image = cv2.resize(re_image, (size, size))

        # YOLOの結果を描画
        img = image[:, :, ::-1]
        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(device)
        img = img.half() if stick_model.fp16 else img.float()
        img /= 255
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = stick_model(img, augment=False, visualize=False)
        pred = non_max_suppression(pred, 0.25, 0.45, None, False, max_det=1000)

        red_point = []
        blue_point = []
        green_point = []
        yellow_point = []

        for det in pred:
            if len(det):
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], (640, 640)).round()

                for *xyxy, conf, cls in reversed(det):
                    # model_names(dict): 0 red, 1 blue, 2 green, 3 yellow
                    label = f'{model_names[int(cls)]} {conf:.2f}'
                    g_point_x = (int(xyxy[0]) + int(xyxy[2])) / size
                    g_point_y = (int(xyxy[1]) + int(xyxy[3])) / size
                    if int(cls) == 0:
                        # xmin ymin xmax ymax
                        if len(red_point) == 0:
                            red_point.append(g_point_x)
                            red_point.append(g_point_y)
                            red_point.append(0.)
                            cv2.rectangle(image, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255, 0, 0), 2)
                    elif int(cls) == 1:
                        if len(blue_point) == 0:
                            blue_point.append(g_point_x)
                            blue_point.append(g_point_y)
                            blue_point.append(0.)
                            cv2.rectangle(image, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 0, 255), 2)
                    elif int(cls) == 2:
                        if len(green_point) == 0:
                            green_point.append(g_point_x)
                            green_point.append(g_point_y)
                            green_point.append(0.)
                            cv2.rectangle(image, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
                    elif int(cls) == 3:
                        if len(yellow_point) == 0:
                            yellow_point.append(g_point_x)
                            yellow_point.append(g_point_y)
                            yellow_point.append(0.)
                            cv2.rectangle(image, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (299, 244, 17), 2)

        # Mediapipeで手の関節点を検出
        results = hands.process(image)

        # 画像を再び書き込み可能にする
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 検出された手の関節点を描画
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # 関節点の座標をリストに追加
                hand_data = []
                for landmark in hand_landmarks.landmark:
                    point_data = [landmark.x, landmark.y, landmark.z]
                    hand_data.append(point_data)
                if len(red_point) * len(blue_point) * len(green_point) * len(yellow_point) != 0:
                    hand_data.append(red_point)
                    hand_data.append(blue_point)
                    hand_data.append(green_point)
                    hand_data.append(yellow_point)
                # データとラベルを保存
                if len(hand_data) == 25:
                    X.append(hand_data)
                    # 正しいなら0,手が違うなら1,箇所が違うなら2
                    y.append(0)
                    # y.append(1)
                    # y.append(2)

        # 画像を表示
        cv2.imshow("Hand Tracking", image)
        if cv2.waitKey(5) & 0xFF == 27:  # ESCキーで終了
            break

with open("X.txt", mode="a") as fx:
    for xitem in X:
        fx.write(f"{xitem}\n")
with open("Y.txt", mode="a") as fy:
    for yitem in y:
        fy.write(f"{yitem}\n")

cap.release()
cv2.destroyAllWindows()
