import os # 文件模块
import tensorflow as tf
import cv2 # 图像处理
import numpy as np
import recognition

data_folder_name = '..\\data'
data_path_name = 'cv'
pic_path_name = 'pic'
cv_path_name = 'fer2013'
csv_file_name = 'fer2013.csv'
ckpt_name = 'cnn_emotion_classifier.ckpt'
model_path_name = 'cnn_inuse'
casc_name = 'haarcascade_frontalface_alt.xml' # Opencv级联分类器
cv_path = os.path.join(data_folder_name, data_path_name, cv_path_name)
csv_path = os.path.join(cv_path, csv_file_name)


channel = 1
default_height = 48
default_width = 48
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']


def main(argv):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 60)
    while cap.isOpened():
        ok, image = cap.read()
        faces, img_gray, img = recognition.face_detect_camera(image)  #识别人脸并转化为灰度图像
        spb = img.shape   #显示尺寸：  宽度，高度，通道数
        sp = img_gray.shape
        height = sp[0]
        width = sp[1]
        size = 600
        emotion_pre_dict = {}
        face_exists = 0
        cv2.imshow('Video', image)
        if len(faces):
            if cv2.waitKey(1) & 0xFF == ord('g'):
                for (x, y, w, h) in faces:
                    face_exists = 1
                    face_img_gray = img_gray[y:y + h, x:x + w]
                    results_sum = recognition.predict_emotion(face_img_gray)  # face_img_gray
                    for i, emotion_pre in enumerate(results_sum):
                        emotion_pre_dict[emotion_labels[i]] = emotion_pre
                    # 输出所有情绪的概率
                    print(emotion_pre_dict)
                    label = np.argmax(results_sum)
                    emo = emotion_labels[int(label)]
                    print('Emotion : ', emo)
                    # 输出最大概率的情绪
                    # 使框的大小适应各种像素的照片
                    with open('../data/cv/1.txt', 'w') as f:
                        f.write(str(emotion_pre_dict)+'\n')
                        f.write(str('Emotion : '+str(emo)))
                    #将所有情绪概率存入文件中，以便后期可视化
                    t_size = 2
                    ww = int(spb[0] * t_size / 300)
                    www = int((w + 10) * t_size / 100)
                    www_s = int((w + 20) * t_size / 100) * 2 / 5
                    # 用方框勾画出脸部位置
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), ww)
                    cv2.putText(img, emo, (x + 2, y + h - 2), cv2.FONT_HERSHEY_SIMPLEX,
                            www_s, (255, 0, 255), thickness=www, lineType=1)
                    #框出人脸并将最大概率的情绪名称注明在图像上
                    # img_gray full face     face_img_gray part of face
                if face_exists:
                    cv2.imwrite('../data/cv/pic/test.jpg',img,[int(cv2.IMWRITE_JPEG_QUALITY),70])
                    cv2.destroyAllWindows()
                    cap.release()
                    return True
                    #将识别处理后的图像保存，以便后期显示在系统界面上


if __name__ == '__main__':
    tf.app.run()

