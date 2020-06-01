import os # 文件模块
import tensorflow as tf
import cv2 # 图像处理
import numpy as np

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
ckpt_path = os.path.join(data_folder_name, data_path_name, ckpt_name)
casc_path = os.path.join(data_folder_name, data_path_name, casc_name)
pic_path = os.path.join(data_folder_name, data_path_name, pic_path_name)


channel = 1
default_height = 48
default_width = 48
confusion_matrix = False
use_advanced_method = True
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
num_class = len(emotion_labels)
y_=[]

# config=tf.ConfigProto(log_device_placement=True)
sess = tf.Session()

saver = tf.train.import_meta_graph(ckpt_path+'.meta') # 加载模型结构
saver.restore(sess, ckpt_path)   # 指定目录就可以恢复所有变量信息
graph = tf.get_default_graph()
name = [n.name for n in graph.as_graph_def().node]
print(name)
x_input = graph.get_tensor_by_name('x_input:0')
dropout = graph.get_tensor_by_name('dropout:0')
logits = graph.get_tensor_by_name('project/output/logits:0')

#图像优化
def advance_image(images_):
    rsz_img = []
    rsz_imgs = []
    for image_ in images_:
        rsz_img.append(image_)
    rsz_imgs.append(np.array(rsz_img))
    rsz_img = []
    for image_ in images_:
        rsz_img.append(np.reshape(cv2.resize(image_[2:45, :], (default_height, default_width)),
                                  [default_height, default_width, channel]))
    rsz_imgs.append(np.array(rsz_img))
    rsz_img = []
    for image_ in images_:
        rsz_img.append(np.reshape(cv2.resize(cv2.flip(image_, 1), (default_height, default_width)),
                                  [default_height, default_width, channel]))
    rsz_imgs.append(np.array(rsz_img))
    return rsz_imgs

#进行人脸识别（单脸）
def produce_result(images_):
    images_ = np.multiply(np.array(images_), 1. / 255)
    if use_advanced_method:
        rsz_imgs = advance_image(images_)
    else:
        rsz_imgs = [images_]
    pred_logits_ = []
    for rsz_img in rsz_imgs:
        pred_logits_.append(sess.run(tf.nn.softmax(logits), {x_input: rsz_img, dropout: 1.0}))
    return np.sum(pred_logits_, axis=0)

#进行人脸识别（多脸）
def produce_results(images_):
    results = []
    pred_logits_ = produce_result(images_)
    pred_logits_list_ = np.array(np.reshape(np.argmax(pred_logits_, axis=1), [-1])).tolist()
    for num in range(num_class):
        results.append(pred_logits_list_.count(num))
    result_decimals = np.around(np.array(results) / len(images_), decimals=3)
    return results, result_decimals

#创建混淆矩阵
def produce_confusion_matrix(images_list_, total_num_):
    total = []
    total_decimals = []
    for ii, images_ in enumerate(images_list_):
        results, result_decimals = produce_results(images_)
        total.append(results)
        total_decimals.append(result_decimals)
        print(results, ii, ":", result_decimals[ii])
        print(result_decimals)
    sum = 0
    for i_ in range(num_class):
        sum += total[i_][i_]
    acc=sum * 100. / total_num_
    print('acc: {:.3f} %'.format(sum * 100. / total_num_))
    print('Using ', ckpt_name)

#转换图片为48*48
def predict_emotion(image_):
    image_ = cv2.resize(image_, (default_height, default_width))
    image_ = np.reshape(image_, [-1, default_height, default_width, channel])
    return produce_result(image_)[0]



#识别人脸并转化为灰度图像
def face_detect_pic(image_path, casc_path_=casc_path):
    if os.path.isfile(casc_path_):
        face_casccade_ = cv2.CascadeClassifier(casc_path_)
        img_ = cv2.imread(image_path) #读入图像
        img_gray_ = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY) #彩色图像转为灰度图像
        # face detection
        # 利用Opencv级联分类器检测脸部
        faces = face_casccade_.detectMultiScale(
            img_gray_,
            scaleFactor=1.1,
            minNeighbors=1,
            minSize=(30, 30),
        )
        return faces, img_gray_, img_
    else:
        print("There is no {} in {}".format(casc_name, casc_path_))


#识别人脸并转化为灰度图像
def face_detect_camera(img_, casc_path_=casc_path):
    if os.path.isfile(casc_path_):
        face_casccade_ = cv2.CascadeClassifier(casc_path_)
        img_gray_ = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY) #彩色图像转为灰度图像
        # face detection
        # 利用Opencv级联分类器检测脸部
        faces = face_casccade_.detectMultiScale(
            img_gray_,
            scaleFactor=1.1,
            minNeighbors=1,
            minSize=(30, 30),
        )
        return faces, img_gray_, img_
    else:
        print("There is no {} in {}".format(casc_name, casc_path_))
