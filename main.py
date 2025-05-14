import cv2
import mediapipe as mp
import numpy as np
import joblib
import torch
import pathlib
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device
import time

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# 保存したmodelをインポート
model = joblib.load("rforest.joblib")
scaler = joblib.load("scaler.joblib")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils


def is_correct_chopstick_grip(hand_data):
    pre_X = np.array(hand_data).reshape(25, 3)
    pre_X = pre_X - pre_X[0]
    x = []
    X = np.zeros(18)
    x_list = np.zeros((5, 3))
    for j in range(5):
        x_list[0] = pre_X[0, :]
        x_list[1] = pre_X[4 * j + 1, :]
        x_list[2] = pre_X[4 * j + 2, :]
        x_list[3] = pre_X[4 * j + 3, :]
        x_list[4] = pre_X[4 * j + 4, :]
        if j == 0:
            x.append(
                np.dot(x_list[0] - x_list[2], x_list[3] - x_list[2]) / np.linalg.norm(x_list[0] - x_list[2]) /
                np.linalg.norm(x_list[3] - x_list[2]))
        else:
            x.append(
                np.dot(x_list[0] - x_list[1], x_list[2] - x_list[1]) / np.linalg.norm(x_list[0] - x_list[1]) /
                np.linalg.norm(x_list[2] - x_list[1]))
            x.append(
                np.dot(x_list[1] - x_list[2], x_list[3] - x_list[2]) / np.linalg.norm(x_list[1] - x_list[2]) /
                np.linalg.norm(x_list[3] - x_list[2]))
        x.append(
            np.dot(x_list[2] - x_list[3], x_list[4] - x_list[3]) / np.linalg.norm(x_list[4] - x_list[3]) /
            np.linalg.norm(x_list[2] - x_list[3]))
    x_5 = (pre_X[21, :2] + pre_X[22, :2]) / 2
    x_6 = (pre_X[23, :2] + pre_X[24, :2]) / 2
    x.append(
        np.dot(pre_X[4, :2] - x_5, pre_X[8, :2] - x_5) / np.linalg.norm(pre_X[4, :2] - x_5) /
        np.linalg.norm(pre_X[8, :2] - x_5))
    x.append(
        np.dot(pre_X[8, :2] - x_5, pre_X[12, :2] - x_5) / np.linalg.norm(pre_X[8, :2] - x_5) /
        np.linalg.norm(pre_X[12, :2] - x_5))
    x.append(
        np.dot(pre_X[12, :2] - x_6, pre_X[16, :2] - x_6) / np.linalg.norm(pre_X[12, :2] - x_6) /
        np.linalg.norm(pre_X[12, :2] - x_6))

    for j in range(len(x)):
        X[j] = x[j]
    X = X.reshape(1, -1)
    test_X = scaler.transform(X)
    pred = model.predict(test_X)

    return pred


def main():
    model_path = "chopstick.pt"

    device = select_device("")
    stick_model = DetectMultiBackend(model_path, device=device)
    model_names = stick_model.names if hasattr(stick_model, "names") else stick_model.module.names

    videopath = "chopsticks-hold/movie/test-mix/test.mp4"

    cap = cv2.VideoCapture(videopath)

    while cap.isOpened():

        start_time = time.time()

        success, image = cap.read()
        if not success:
            break

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # 画像のパディング
        origin_height, origin_width = image.shape[:2]
        size = 640

        if origin_width >= origin_height:
            re_image = cv2.copyMakeBorder(image, 0, (origin_width - origin_height), 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        else:
            re_image = cv2.copyMakeBorder(image, 0, 0, 0, (origin_height - origin_width), cv2.BORDER_CONSTANT, value=[0, 0, 0])

        image = cv2.resize(re_image, (size, size))

        # YOLOを実行
        img = image[:, :, ::-1]
        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(device)
        img = img.half() if stick_model.fp16 else img.float()  # uint8 to fp16/32
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

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # ランドマークを描画
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                hand_data = []
                for landmark in hand_landmarks.landmark:
                    point_data = [landmark.x, landmark.y, landmark.z]
                    hand_data.append(point_data)
                if len(red_point) * len(blue_point) * len(green_point) * len(yellow_point) != 0:
                    hand_data.append(red_point)
                    hand_data.append(blue_point)
                    hand_data.append(green_point)
                    hand_data.append(yellow_point)

                if len(hand_data) == 25:
                    # 持ち方を判定
                    if is_correct_chopstick_grip(hand_data) == 0:
                        cv2.putText(image, "Correct Grip", (10, 30), cv2.FONT_ITALIC, 1, (0, 255, 0), 2,
                                    cv2.LINE_AA)
                    elif is_correct_chopstick_grip(hand_data) == 1:
                        cv2.putText(image, "Incorrect Grip", (10, 30), cv2.FONT_ITALIC, 1, (0, 0, 255), 2,
                                    cv2.LINE_AA)
                    else:
                        cv2.putText(image, "Missplace", (10, 30), cv2.FONT_ITALIC, 1, (0, 0, 255), 2,
                                    cv2.LINE_AA)
        end_time = time.time()

        print(end_time - start_time)

        # 画像を表示
        cv2.imshow("Chopstick Grip Detection", image)
        
        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
