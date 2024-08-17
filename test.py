
import pyautogui

from PIL import ImageGrab
import tkinter as tk
import math
import re
import html
import xml.etree.ElementTree as ET
import numpy as np


from pynput.mouse import Listener, Button

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from paddleocr import PaddleOCR, draw_ocr


def parse_xml(file_path):
    # 解析XML文件
    tree = ET.parse(file_path)
    root = tree.getroot()

    # 提取contentList中的英文句子和对应的汉语翻译
    english_sentences = []
    chinese_translations = []
    for content in root.findall('content'):
        text = content.text
        if text:
            text = html.unescape(text)
            parts = text.split('<br>')
            if len(parts) == 2:
                english_sentences.append(parts[0].strip())
                chinese_translations.append(parts[1].strip())
    return english_sentences, chinese_translations


# 解析XML文件中的英文句子和对应汉语翻译
xml_file_path = "output.xml"
english_sentences, chinese_translations = parse_xml(xml_file_path)

# 使用TF-IDF向量化
vectorizer = TfidfVectorizer().fit(english_sentences)
english_vectors = vectorizer.transform(english_sentences)


ocr = PaddleOCR()  # need to run only once to download and load model into memory






def get_most_similar_sentence(ocr_text):

    ocr_vector = vectorizer.transform([ocr_text])
    
    # 计算OCR句子与XML中的句子的余弦相似度
    similarities = cosine_similarity(ocr_vector, english_vectors).flatten()
    
    # 找到相似度最高的句子
    most_similar_idx = np.argmax(similarities)
    highest_similarity = similarities[most_similar_idx]
    
    return most_similar_idx, highest_similarity





def filter_text(text):
    # 去除开头和结尾的空白字符
    text = text.strip()
    
    # 使用while循环不断删除开头的不需要的字符，直到不再匹配
    while True:
        # 匹配以下开头模式：序号、句号、尖括号、以及 [ILLITHID] [WISDOM] 这种标签模式
        match = re.match(r"^([\s0-9\.\<\>\[\]A-Z]+[:\s]*)", text)
        if match:
            # 如果匹配成功，删除匹配部分
            text = text[match.end():]
        else:
            break
    return text

def find_nearest_sentence(mouse_x, mouse_y, ocr_results):
    min_distance = float('inf')
    nearest_sentence = None

    if ocr_results:
        for result in ocr_results:
            # 获取OCR检测到的文本位置和内容
            position, (text, confidence) = result

            # 计算位置的中心点
            center_x = (position[0][0] + position[2][0]) / 2
            center_y = (position[0][1] + position[2][1]) / 2

            # 计算鼠标位置与中心点的欧几里得距离
            distance = math.sqrt((mouse_x - center_x) ** 2 + (mouse_y - center_y) ** 2)

            if distance < min_distance:
                min_distance = distance
                nearest_sentence = text

    return nearest_sentence





# 全局变量，用于控制弹框的状态
popup_window = None



def on_click(x, y, button, pressed):
    global popup_window

    if button == Button.x2 and pressed and popup_window == None:
        # 获取鼠标当前位置
        mouse_x, mouse_y = pyautogui.position()

        # 计算截图区域的坐标 (左上角和右下角)
        left = mouse_x - 1000
        top = mouse_y - 50
        right = mouse_x + 1000
        bottom = mouse_y + 50

        # 截取屏幕指定区域
        img = ImageGrab.grab(bbox=(left, top, right, bottom))
        img_array = np.array(img)

        result = ocr.ocr(img_array, cls=False)
            

        # 找到与鼠标最近的句子
        nearest_sentence = find_nearest_sentence(mouse_x, mouse_y, result[0])

        # 显示结果
        if nearest_sentence:
            # 过滤OCR结果
            filtered_sentence = filter_text(nearest_sentence)

            # 找到最相似的句子及其对应的汉语翻译
            most_similar_idx, similarity = get_most_similar_sentence(filtered_sentence)

            if most_similar_idx is not None:
                most_similar_sentence = english_sentences[most_similar_idx]
                corresponding_translation = chinese_translations[most_similar_idx]
                
                print(f"最相似的英文句子: {most_similar_sentence}")
                print(f"相似度: {similarity:.2f}")
                print(f"对应的汉语翻译: {corresponding_translation}")

                # 显示弹框
                show_text_popup(corresponding_translation, mouse_x, mouse_y)
                print("弹框已显示")
    elif button == Button.x2 and pressed and popup_window is not None:
        # 关闭弹框
        popup_window.destroy()
        popup_window = None
        print("弹框已关闭")



 


def show_text_popup(text, x, y):
    global popup_window

    # 创建一个新的Tk窗口
    popup_window = tk.Tk()
    popup_window.overrideredirect(1)  # 去掉窗口边框
    popup_window.geometry(f"+{x}+{y-100}")  # 设置窗口位置为鼠标当前位置的上方100像素
    popup_window.attributes("-topmost", True)  # 设置窗口始终在最上层

    # 在窗口内显示文本
    #label = tk.Label(root, text=text, background="white", relief="solid")
    # label = tk.Label(popup_window, text=text, background="white", foreground="black", 
    #                  relief="solid", borderwidth=2, padx=10, pady=5, font=("Arial", 18))

    label = tk.Label(popup_window, text=text,padx=2,pady=2,borderwidth=2, font=("微软雅黑", 22))
    label.pack()

    # 开始窗口的事件循环
    popup_window.update()





# 启动监听
with Listener(on_click=on_click) as listener:
    listener.join()
