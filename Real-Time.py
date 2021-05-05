import numpy as np
import cv2
import tkinter as tk
from tkinter import messagebox as msg
from PIL import Image


class FaceDetection:
    def __init__(self):
        pass

    @staticmethod
    def CheckUsingPersonalInformation():  # 개인정보 활용 동의 여부를 물어봅니다.
        MsgBox = msg.askquestion('개인정보 활용 동의', '카메라를 통해 얼굴을 캡처합니다. 동의하시겠습니까?')
        if MsgBox == 'no':
            msg.showinfo('개인정보 활용 동의', '프로그램을 종료합니다.')
            MsgBox.destroy()
        else:
            msg.showinfo('개인정보 활용 동의', '카메라를 통해 캡처를 시작합니다.')
            # StartCapture() <-- 이런식으로 ?

    def ImgTrim(self, img, x, y, w, h):
        """
        이미지 파란 박스에 해당하는 부분만 잘라 저장하기
        :param img: 저장할 이미지
        :param x: x 좌표
        :param y: y 좌표
        :param w: width
        :param h: height
        :return: img_trim
        """
        x_trim = x;
        y_trim = y;
        w_trim = w;
        h_trim = h;
        saved_img = img[y:y_trim + h_trim, x:x_trim + w_trim]
        face = cv2.resize(saved_img, dsize=(512, 512), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite('face_img.png', face)
        return face

    def NotUsingResizeImgTrim(self, img, x, y, w, h):
        """
        이미지 파란 박스에 해당하는 부분만 잘라 저장하기
        :param img: 저장할 이미지
        :param x: x 좌표
        :param y: y 좌표
        :param w: width
        :param h: height
        :return: img_trim
        """
        x_trim = x;
        y_trim = y;
        w_trim = w;
        h_trim = h;
        saved_img = img[y:y_trim + h_trim, x:x_trim + w_trim]
        cv2.imwrite('face_img_2.png', saved_img)
        return saved_img

    def StartFaceCapture(self):  # 얼굴 캡처를 시작함
        """
        영상에서 얼굴 캡처를 시작하여, origin_img를 만듬
        :return: origin_img
        """
        # 얼굴 위치 잡을 수 있도록 해주는 xml
        xml = 'haarcascades/haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(xml)

        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # 노트북 웹캠을 카메라로 사용
        cap.set(3, 640)  # 너비
        cap.set(4, 480)  # 높이

        # fourcc = cv2.VideoWriter_fourcc(*'XVID') # 캡처한 영상을 저장할 때 씀
        # out = cv2.VideoWriter('result/output.mp4', fourcc, 20.0, (640, 480)) # 캡처한 영상을 저장하기 위함

        while True:
            ret, frame = cap.read()  # 비디오의 프레임을 한 프레임씩 읽기
            frame = cv2.flip(frame, 1)  # 좌우 대칭 프레임 생성
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 프레임을 흑백으로 지정
            # out.write(frame) # 캡처한 영상을 저장하기 위함

            faces = face_cascade.detectMultiScale(gray, 1.05, 5)

            if len(faces):
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x - 30, y - 45), (x + w + 30, y + h + 30), (255, 0, 0), 1)
                    self.ImgTrim(frame, x - 25, y - 45, w + 50, h + 70)
                    self.NotUsingResizeImgTrim(frame, x - 25, y - 45, w + 50, h + 70)

                cv2.imshow('result', frame)

                k = cv2.waitKey(30) & 0xff

                if k == 27:  # Esc 키를 누르면 종료
                    break

        cap.release()
        cv2.destroyAllWindows()

    def GenerateImageUsingMichiGAN(origin_img, target_hairstyle_mask):
        """
        원본 이미지와, 타겟 헤어스타일 마스크를 이용해 사진 만들기
        :param origin_img: 원본 이미지
        :param target_hairstyle_mask: 타겟 마스크 (바꾸고 싶은 헤어스타일)
        :return: output_img
        """
        pass

    def ShowResult(output_img):
        """
        MichiGAN으로부터 도출된 output_img을 이쁘게 보여주기 -> (원본 이미지, 변경된 이미지) 이렇게 보여준다던가..
        :param output_img: 생성된 사진
        :return: final_output_img
        """
        pass

    def ShowResultTest(self):
        image = cv2.imread("face_img.png", cv2.IMREAD_COLOR)

        cv2.imshow('Face', image)  # 저장된 사진 보내주기

        cv2.waitKey(0)  # 사용자 키 입력 기다림
        cv2.destroyAllWindows()


a = FaceDetection()
a.StartFaceCapture()
