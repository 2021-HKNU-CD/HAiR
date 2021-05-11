import numpy as np
import cv2
import tkinter as tk
from tkinter import messagebox as msg


def CheckUsingPersonalInformation() -> bool:  # 개인정보 활용 동의 여부를 물어봅니다.
    MsgBox = msg.askquestion('개인정보 활용 동의', '카메라를 통해 얼굴을 캡처합니다. 동의하시겠습니까?')
    if MsgBox == 'no':
        msg.showinfo('개인정보 활용 동의', '프로그램을 종료합니다.')
        MsgBox.destroy()
        return False
    else:
        msg.showinfo('개인정보 활용 동의', '카메라를 통해 캡처를 시작합니다.')
        return True

def trim( img: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    x_trim = x
    y_trim = y
    w_trim = w
    h_trim = h
    saved_img = img[y:y_trim + h_trim, x:x_trim + w_trim]
    face = cv2.resize(saved_img, dsize=(512, 512),
                        interpolation=cv2.INTER_LINEAR)
    # cv2.imwrite('face_img.png', face)
    return face

def trim_without_resize( img: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    x_trim = x
    y_trim = y
    w_trim = w
    h_trim = h
    saved_img = img[y:y_trim + h_trim, x:x_trim + w_trim]
    # cv2.imwrite('face_img_2.png', saved_img)
    return saved_img


class FaceDetection():
    def __init__(self):
        pass

    def capture(self):
        face_pattern = 'haarcascades/haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(face_pattern)

        cap = cv2.VideoCapture(1)
        cap.set(3, 1920)
        cap.set(4, 1080)

        while True:
            ret, frame = cap.read()  # 비디오의 프레임을 한 프레임씩 읽기
            frame = cv2.flip(frame, 1)  # 좌우 대칭 프레임 생성
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 프레임을 흑백으로 지정
            # out.write(frame) # 캡처한 영상을 저장하기 위함

            faces = face_cascade.detectMultiScale(gray, 1.05, 5)

            if len(faces):
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x - 30, y - 45), (x + w + 30, y + h + 30), (255, 0, 0), 1)
                    trim(frame, x - 25, y - 45, w + 50, h + 70)
                    trim_without_resize(frame, x - 25, y - 45, w + 50, h + 70)

                cv2.imshow('result', frame)

                k = cv2.waitKey(30) & 0xff

                if k == 27:  # Esc 키를 누르면 종료
                    break

        cap.release()
        cv2.destroyAllWindows()
            
    def ShowResultTest(self):
        image = cv2.imread("face_img.png", cv2.IMREAD_COLOR)

        cv2.imshow('Face', image)  # 저장된 사진 보내주기

        cv2.waitKey(0)  # 사용자 키 입력 기다림
        cv2.destroyAllWindows()

a = FaceDetection()
a.capture()