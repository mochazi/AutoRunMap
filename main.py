#coding=utf-8
import time,os,sys,random,shutil,pywinauto,ctypes,ddddocr,threading,multiprocessing,pydirectinput
import numpy as np
import pyautogui,pygetwindow
import cv2
import logging
import traceback
from concurrent.futures import ThreadPoolExecutor,as_completed
from copy import deepcopy
from ctypes import c_char_p

INFO = None
VERSION = 'v1.0'
LOCATION = None
LOCK = threading.Lock()
WINDOWS_NAME = None

# 解决pyinstaller打包找不到静态文件问题
def resource_path(relative_path, debug=False):
    """ Get absolute path to resource, works for dev and for PyInstaller """

    if debug:
        base_path = os.path.abspath("./temp/")
        return os.path.join(base_path, relative_path)
    else:
        try:
            # PyInstaller creates a temp folder and stores path in _MEIPASS
            base_path = sys._MEIPASS
        except Exception:
            base_path = os.path.abspath(".")

        return os.path.join(base_path, relative_path)

# 检查游戏窗口是否存在
def check_window_exist(window_title):
    for window in pygetwindow.getAllTitles():
        if window.lower() == window_title.lower():
            return True
    return False

# 确保存在文件夹
if not os.path.exists('temp'):
    os.mkdir('temp')

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)

# Create a FileHandler and set its level to ERROR (or higher)
file_handler = logging.FileHandler(resource_path('error.log', debug=True))
file_handler.setLevel(logging.ERROR)

# Create a Formatter for formatting the log messages
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add the FileHandler to the logger
logger.addHandler(file_handler)


'''
    解决pyinstaller打包找不到模型问题
    修改venv/Lib/site-packages/ddddocr/__init__.py
    if ocr:
        if not beta:
            self.__graph_path = resource_path('model/common_old.onnx')
'''
OCR = ddddocr.DdddOcr(ocr=True,show_ad=False)


# 进程退出标记
PROCESS_FLAG = multiprocessing.Value('b', True)


class NumberOCR():

    def __init__(self):
        self.ocr = ddddocr.DdddOcr(show_ad=False)
        # 阈值
        self.lowerb = [100,110,120,95]

    # 读取图像，解决imread不能读取中文路径的问题
    @staticmethod
    def cv_imread(filePath):
        # 核心就是下面这句，一般直接用这句就行，直接把图片转为mat数据
        cv_img=cv2.imdecode(np.fromfile(filePath,dtype=np.uint8),-1)
        # imdecode读取的是rgb，如果后续需要opencv处理的话，需要转换成bgr，转换后图片颜色会变化
        # cv_img=cv2.cvtColor(cv_img,cv2.COLOR_RGB2BGR)
        return cv_img
    
    # 拆分数字
    @staticmethod
    def split_number(num):
        num_str = str(num)
        digits = [int(digit) for digit in num_str]
        return digits

    # 判断是不是数字
    @staticmethod
    def is_number(str):
        return str.isdigit()

    # 验证码图片（预处理）
    @staticmethod
    def clear_images(lowerb):
        
        img = NumberOCR.cv_imread(f'./temp/验证码.jpg')
        img2 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img2 = cv2.inRange(img2, lowerb=lowerb, upperb=255)
        # ret, img2 = cv2.threshold(img2, 120, 255, cv2.THRESH_BINARY)
        img2 = NumberOCR.operate_img(img2, 1)
        cv2.imencode('.jpg', img2)[1].tofile(f'./temp/验证码降噪.jpg')

    # 识别验证码
    @staticmethod
    def working(lowerb):
        try:
            
            with LOCK:
                NumberOCR.clear_images(lowerb)

            with open(f'./temp/验证码降噪.jpg', 'rb') as f:
                image = f.read()

                res = OCR.classification(image)
                # print(res)
                return lowerb,res
            
        except:
            print(traceback.format_exc())
            logger.error(traceback.format_exc())
    
    # 启动线程池
    @staticmethod
    def ocr_working():

        lowerb_list = [100,110,120,95]

        try:
            with ThreadPoolExecutor(max_workers=4) as t:
                work_list = [t.submit(NumberOCR.working,lowerb) for lowerb in lowerb_list]
                    
                # 获取任务结果
                results = {}
                for future in as_completed(work_list):
                    try:
                        if future.result():
                            lowerb,result = future.result()

                            if result.isdigit():
                                results[lowerb] = result
                    except:
                        print(traceback.format_exc())
                        logger.error(traceback.format_exc())
                
                code = set()

                for lowerb in lowerb_list:
                    if results.get(lowerb):
                        code.add(results.get(lowerb))

                code = int(list(code)[0]) if list(code) else None
                # print(code)
                if code:
                    return NumberOCR.split_number(code)
                else:
                    return code
        except:
            print(traceback.format_exc())
            logger.error(traceback.format_exc())


    # 计算邻域非白色个数
    @staticmethod
    def calculate_noise_count(img_obj, w, h):
        """
        计算邻域非白色的个数
        Args:
            img_obj: img obj
            w: width
            h: height
        Returns:
            count (int)
        """
        count = 0
        width, height = img_obj.shape
        for _w_ in [w - 1, w, w + 1]:
            for _h_ in [h - 1, h, h + 1]:
                if _w_ > width - 1:
                    continue
                if _h_ > height - 1:
                    continue
                if _w_ == w and _h_ == h:
                    continue
                if (img_obj[_w_, _h_] < 233) or (img_obj[_w_, _h_] < 233) or (img_obj[_w_, _h_] < 233):
                    count += 1

        return count


    # k邻域降噪
    @staticmethod
    def operate_img(img,k):
        w,h = img.shape
        # 从高度开始遍历
        for _w in range(w):
            # 遍历宽度
            for _h in range(h):
                if _h != 0 and _w != 0 and _w < w-1 and _h < h-1:
                    if NumberOCR.calculate_noise_count(img, _w, _h) < k:
                        img.itemset((_w,_h),255)
                        img.itemset((_w, _h), 255)
                        img.itemset((_w, _h), 255)
        return img



class AutoRunMap():

    # 读取图像，解决imread不能读取中文路径的问题
    def cv_imread(self, filePath):
        # 核心就是下面这句，一般直接用这句就行，直接把图片转为mat数据
        cv_img=cv2.imdecode(np.fromfile(filePath,dtype=np.uint8),-1)
        # imdecode读取的是rgb，如果后续需要opencv处理的话，需要转换成bgr，转换后图片颜色会变化
        # cv_img=cv2.cvtColor(cv_img,cv2.COLOR_RGB2BGR)
        return cv_img

    def get_windows_location(self):

        try:
            time.sleep(0.2)
            app = pywinauto.Application().connect(title_re=WINDOWS_NAME)
            
            # 获取主窗口
            main_window = app.window(title=WINDOWS_NAME)

            # 将窗口置顶
            main_window.set_focus()
            main_window.topmost = True

            window = app.top_window()
            left, right, top, down  = window.rectangle().left, window.rectangle().right, window.rectangle().top, window.rectangle().bottom
            # print(f"The window position is ({left}, {right}, {top}, {down})")
            return left, right, top, down
        except pywinauto.findwindows.ElementNotFoundError:
            print(traceback.format_exc())
            # INFO = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S ') + '没有匹配目标, 请确保游戏窗口没有脱离显示器屏幕'
            # logger.error(traceback.format_exc())
        except TypeError:
            print(traceback.format_exc())


    def get_keypoint_bounds(self,kp):
        # 获取关键点的中心坐标和直径大小
        x, y = kp.pt
        diameter = kp.size

        # 计算关键点的左上角和右下角坐标
        x1 = int(x - diameter/2)
        y1 = int(y - diameter/2)
        x2 = int(x + diameter/2)
        y2 = int(y + diameter/2)

        return x1, y1, x2, y2


    # 获取大图里面的小图坐标（简单匹配，计算量少，速度快）
    def get_simple_xy(self, img_model_path,verification_code=False,name=None, show=False, output=True):
        """
        用来判定游戏画面的点击坐标
        :param img_model_path:用来检测的图片
        :return:以元组形式返回检测到的区域中心的坐标
        """

        # 确保存在文件夹
        if not os.path.exists('temp'):
            os.mkdir('temp')

        # 将图片截图并且保存
        pyautogui.screenshot().save("./temp/screenshot.jpg")
        # 待读取图像
        img = self.cv_imread("./temp/screenshot.jpg")

        # 裁剪图片
        img, cropped_X, cropped_Y = self.cropped_images(img)
        if cropped_X == False:
            return False, False

        # 图像模板
        img_terminal = self.cv_imread(img_model_path)
        # 读取模板的高度宽度和通道数
        height, width, channel = img_terminal.shape
        # 使用matchTemplate进行模板匹配（标准平方差匹配）
        result = cv2.matchTemplate(img, img_terminal, cv2.TM_SQDIFF_NORMED)

        # -----------------匹配最优对象-----------------

        # 解析出匹配区域的左上角图标
        upper_left = cv2.minMaxLoc(result)[2]
        # 计算出匹配区域右下角图标（左上角坐标加上模板的长宽即可得到）
        lower_right = (upper_left[0] + width, upper_left[1] + height)
        # 计算坐标的平均值并将其返回
        avg = (int((upper_left[0] + lower_right[0]) / 2) + cropped_X, int((upper_left[1] + lower_right[1]) / 2) + cropped_Y)

        # 针对验证码特殊处理
        if verification_code:
            return (upper_left,lower_right),img
        
        # 展示图片
        if show:
            cv2.rectangle(img, upper_left, lower_right, (0, 0, 255), 2)
            cv2.imshow('img', img)
            cv2.waitKey()

        # 输出图片
        if output:
            cv2.rectangle(img, upper_left, lower_right, (0, 0, 255), 2)
            if name != None:
                # 中文路径输出
                cv2.imencode('.jpg', img)[1].tofile(f'./temp/{name}.jpg')
            else:
                cv2.imwrite(f'./temp/get_xy.jpg', img)
        # ---------------------------------------------
        return avg,img
    
    # 获取大图里面的小图坐标（精准匹配，针对简单匹配无法解决的问题，计算量多，性能慢）
    def get_precise_xy(self, img_model_path,check_reday_or_start=False,name=None,show=False, output=True):

        """
        用来判定游戏画面的点击坐标
        :param img_model_path:用来检测的图片
        :return:以元组形式返回检测到的区域中心的坐标
        """
        # 确保存在文件夹
        if not os.path.exists('temp'):
            os.mkdir('temp')

        # 将图片截图并且保存
        pyautogui.screenshot().save("./temp/screenshot.jpg")

        # 待读取图像
        img_rgb = cv2.imread("./temp/screenshot.jpg")
        img_rgb, cropped_X, cropped_Y = self.cropped_images(img_rgb)
        if cropped_X == False:
            return False,False

        # 图像模板
        template = self.cv_imread(img_model_path)

        template_copy = None
        img_rgb_copy = None
         # 针对「游戏准备」「游戏开始」仅计算左边45%
        if check_reday_or_start:
            
            template_copy = deepcopy(template)
            img_rgb_copy = deepcopy(img_rgb)

            # ------------ 处理原始图像 （只要右下角）------------
            # 计算图像高度的35%
            img_height = img_rgb.shape[0]
            bottom_height = int(img_height * 0.35)

            # 裁剪下面35%的图像
            img_rgb = img_rgb[img_height - bottom_height:, :]

            # 计算图像宽度的35%
            img_width = img_rgb.shape[1]
            right_width = int(img_width * 0.35)

            # 裁剪右边35%的图像
            img_rgb = img_rgb[:, img_width - right_width:]

            # ------------ 处理模板图像 （只要左边） ------------
            # 计算图像宽度的45%
            template_width = template.shape[1]
            left_width = int(template_width * 0.45)

            # 裁剪左边40%的图像
            template = template[:, :left_width]

        # 将图像转换为灰度图像
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

        # 初始化SIFT检测器
        sift = cv2.SIFT_create()

        # 在模板图像中检测特征点和描述符
        kp1, des1 = sift.detectAndCompute(template_gray, None)

        # 在原始图像中检测特征点和描述符
        kp2, des2 = sift.detectAndCompute(img_gray, None)

        # 初始化FLANN匹配器
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        # 使用FLANN匹配器进行匹配
        matches = flann.knnMatch(des1, des2, k=2)

        # 过滤匹配点
        good_matches = []
        for number in [0.3]:
            for m, n in matches:
                if m.distance < number * n.distance:
                    good_matches.append(m)

        # 绘制匹配结果
        template = template_copy
        img_rgb = img_rgb_copy
        img_matches = cv2.drawMatches(template, kp1, img_rgb, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


        # 获取所有好的匹配点在原始图像中的中心坐标
        sum_x = 0
        sum_y = 0
        for match in good_matches:
            # 获取关键点在原始图像中的索引
            index = match.trainIdx

            # 获取关键点的左上角和右下角坐标
            x1, y1, x2, y2 = self.get_keypoint_bounds(kp2[index])

            # 计算中心坐标
            mid_x = int((x1 + x2) / 2)
            mid_y = int((y1 + y2) / 2)

            # 累加中心坐标
            sum_x += mid_x
            sum_y += mid_y

            # 获取关键点坐标
            x1, y1 = kp1[match.queryIdx].pt
            x2, y2 = kp2[match.trainIdx].pt

            # 绘制红色边框
            cv2.rectangle(img_rgb, (int(x2)-5, int(y2)-5), (int(x2)+5, int(y2)+5), (0, 0, 255), 2)

        # 展示图片
        if show:
            # cv2.rectangle(img, upper_left, lower_right, (0, 0, 255), 2)
            cv2.imshow('img', img_matches)
            cv2.waitKey()

        # 输出图片
        if output:

            if name != None:
                # 中文路径输出
                cv2.imencode('.jpg', img_matches)[1].tofile(f'./temp/{name}.jpg')
            else:
                cv2.imwrite(f'./temp/get_precise_xy.jpg', img_matches)
        # ---------------------------------------------

        # 计算平均中心坐标
        if len(good_matches) == 0:
            return False,False
        else:
            avg_x = int(sum_x / len(good_matches)) + cropped_X
            avg_y = int(sum_y / len(good_matches)) + cropped_Y

            return (avg_x, avg_y), img_matches
        
    # 裁剪图片
    def cropped_images(self, img_rgb):

        try:
            # 获取指定窗口坐标并置顶窗口
            left, right, top, down = self.get_windows_location()

            # ---------------裁剪图片------------------

            # print(img_rgb.shape)

            # cropped = img_rgb[top+300:down-800,left+400:right-300]  # 裁剪坐标为[y0:y1, x0:x1]
            cropped = img_rgb[top:down,left:right]  # 裁剪坐标为[y0:y1, x0:x1]

            height, width = cropped.shape[:2]

            # print((height, width))

            # 计算中间区域的左上角X和Y坐标以及其宽度和高度
            # cropWidth = int(width * 0.55)
            # cropHeight = int(height * 0.9)
            # cropped_X = (width - cropWidth) // 2
            # cropped_Y = (height - cropHeight) // 2
            cropped_X = 0
            cropped_Y = 0

            # print()
            # print((cropWidth, cropHeight))
            # print((cropped_X, cropped_Y))

            # cropped = cropped[cropped_Y:cropped_Y+cropHeight, cropped_X:cropped_X+cropWidth]

            cv2.imwrite("./temp/screenshot_cropped.jpg", cropped)
            img_rgb = cropped

            # 返回裁剪图片，x偏移量，y偏移量
            return img_rgb, cropped_X + left, cropped_Y + top
        except Exception as error:
            print(traceback.format_exc())
            logger.error(traceback.format_exc())
            INFO = '没有匹配目标, 请确保游戏窗口没有脱离显示器屏幕'
            return False, False, False

    # 自动点击
    def auto_click(self, var_avg):
        """
        输入一个元组，自动点击
        :param var_avg: 坐标元组
        :return: None
        """
        self.get_windows_location()
        pyautogui.click(var_avg[0], var_avg[1], button='left')
        time.sleep(0.2)


    # 点击验证码（简单匹配）
    def simple_click_verification_code_images(self, click_list, show=False, output=True):

        global INFO

        # 确保存在文件夹
        if not os.path.exists('temp'):
            os.mkdir('temp')

        img = None
        # 排序（确定左和右）
        for index in click_list:
            
            verification_code_image_path = resource_path(f'images/core/{index}.jpg', debug=True)
            avg,img = self.get_simple_xy(verification_code_image_path)

            # INFO = f"正在点击图片{verification_code_image_path}"
            # print(INFO)

            self.auto_click(avg)
            pyautogui.moveTo(avg[0]+300, avg[1])


        # 展示图片
        if show:
            cv2.imshow('img', img)
            cv2.waitKey()
        
        # 输出图片
        if output:
            cv2.imwrite('./temp/click_verification_code_images.jpg', img)

        if click_list == {}:
            return False
        else:
            INFO = f"点击验证码完毕"
             # print(INFO)
            return True
       
        # -------------------------------------------

    # 裁剪验证码图片
    def cropped_verification_code_images(self,show=False, output=True):

        # 获取验证码坐标
        avg,img_rgb = self.get_simple_xy(resource_path(f'images/core/请框住验证码的位置.jpg',debug=True),verification_code=True)
        # print(avg)

        # ---------------裁剪图片------------------

        upper_left = avg[0]
        lower_right = avg[1]
        # print(img_rgb.shape)

        # cropped = img_rgb[top+300:down-800,left+400:right-300]  # 裁剪坐标为[y0:y1, x0:x1]
        # cropped = img_rgb[top:down,left:right]  # 裁剪坐标为[y0:y1, x0:x1]
        
        # 开始裁剪
        cropped = img_rgb[upper_left[1]:lower_right[1], upper_left[0]:lower_right[0]]

        # height, width = cropped.shape[:2]
        # print((height, width))

        if show:
            cv2.imshow('img', cropped)
            cv2.waitKey()

        if output:
            # 中文路径输出
            cv2.imencode('.jpg', cropped)[1].tofile(f'./temp/验证码.jpg')
            cv2.rectangle(img_rgb, upper_left, lower_right, (0, 0, 255), 2)
            cv2.imwrite("./temp/screenshot_cropped.jpg", img_rgb)

        img_rgb = cropped

        # 返回裁剪图片，状态码
        return img_rgb,True
    
    
    def check_reday_or_start(self,image_path, keys):

        with LOCK:
            print("正在检查「游戏准备」「游戏开始」")
            INFO = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S ') + '正在检查「游戏准备」「游戏开始」'
            self.tk_text_lb4no8xs.insert(END, INFO + "\r\n", 'green')
            self.tk_text_lb4no8xs.see(END)

        avg,_ = self.get_simple_xy(image_path)
        
        if avg:

            if keys == 'F9':
                with LOCK:
                    print("按下「游戏准备」")
                    INFO = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S ') + '按下「游戏准备」'
                    self.tk_text_lb4no8xs.insert(END, INFO + "\r\n", 'green')
                    self.tk_text_lb4no8xs.see(END)
            if keys == 'F10':
                with LOCK:
                    print("按下「游戏开始」")
                    INFO = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S ') + '按下「游戏开始」'
                    self.tk_text_lb4no8xs.insert(END, INFO + "\r\n", 'green')
                    self.tk_text_lb4no8xs.see(END)
            

            pyautogui.press(keys)
        

    

#coding=utf-8
import datetime,threading,PIL,webbrowser,json,subprocess
from time import sleep
from tkinter import Label,Frame,messagebox,Entry,Tk,Text,Button,Toplevel,END
from tkinter import messagebox
from PIL import ImageTk

class WinGUI(Tk):
    def __init__(self):
        super().__init__()
        self.__win()
        self.main_thread = None
        self.up_key = self.__init_up_key()
        self.jump_key = self.__init_jump_key()
        self.tk_text_lb4no8xs = self.__tk_text_lb4no8xs()
        self.tk_button_start = self.__tk_button_start()
        self.tk_button_stop = self.__tk_button_stop()
        self.tk_button_tutorial = self.__tk_button_tutorial()
        self.tk_label_up_key = self.__tk_label_up_key()
        self.tk_label_jump_key = self.__tk_label_jump_key()
        self.tk_label_sprint_time_key = self.__tk_label_sprint_time_key()
        self.tk_label_extra_sprint_time_key = self.__tk_label_extra_sprint_time_key()
        self.tk_entry_up_key = self.__tk_entry_up_key()
        self.tk_entry_jump_key = self.__tk_entry_jump_key()
        self.tk_entry_sprint_time_key = self.__tk_entry_sprint_time_key()
        self.tk_entry_extra_sprint_time_key = self.__tk_entry_extra_sprint_time_key()

    def __init_up_key(self):

        # 确保存在文件夹
        if not os.path.exists('temp'):
            os.mkdir('temp')

        # 复制exe的images文件夹出来
        if os.path.exists('./temp/images'):
            # 如果目标文件夹已经存在，删除它
            # shutil.rmtree('./temp/images')
            pass
        else: 
            shutil.copytree(resource_path('images'), './temp/images')


        if not os.path.exists('./temp/settings.json'):
            with open('./temp/settings.json','w',encoding='utf-8') as f:
                f.write(json.dumps(dict(up_key='up',jump_key='ctrl',sprint_time_key=0.9,extra_sprint_time_key=0.0), ensure_ascii=False))
            return 'up'
        else:
            with open('./temp/settings.json','r',encoding='utf-8') as f:
                up_key = json.loads(f.read()).get('up_key')
                # print(up_key)
                if up_key != None:
                    return up_key
                else:
                    return 'up'
    
    def __init_jump_key(self):

        # 确保存在文件夹
        if not os.path.exists('temp'):
            os.mkdir('temp')

        # 复制exe的images文件夹出来
        if os.path.exists('./temp/images'):
            # 如果目标文件夹已经存在，删除它
            # shutil.rmtree('./temp/images')
            pass
        else: 
            shutil.copytree(resource_path('images'), './temp/images')


        if not os.path.exists('./temp/settings.json'):
            with open('./temp/settings.json','w',encoding='utf-8') as f:
                f.write(json.dumps(dict(up_key='up',jump_key='ctrl',sprint_time_key=0.9,extra_sprint_time_key=0.0), ensure_ascii=False))
            return 'ctrl'
        else:
            with open('./temp/settings.json','r',encoding='utf-8') as f:
                jump_key = json.loads(f.read()).get('jump_key')
                # print(jump_key)
                if jump_key != None:
                    return jump_key
                else:
                    return 'ctrl'
    
    def __init_sprint_time_key(self):

        # 确保存在文件夹
        if not os.path.exists('temp'):
            os.mkdir('temp')

        # 复制exe的images文件夹出来
        if os.path.exists('./temp/images'):
            # 如果目标文件夹已经存在，删除它
            # shutil.rmtree('./temp/images')
            pass
        else: 
            shutil.copytree(resource_path('images'), './temp/images')


        if not os.path.exists('./temp/settings.json'):
            with open('./temp/settings.json','w',encoding='utf-8') as f:
                f.write(json.dumps(dict(up_key='up',jump_key='ctrl',sprint_time_key=0.9,extra_sprint_time_key=0.0), ensure_ascii=False))
            return 0.9
        else:
            with open('./temp/settings.json','r',encoding='utf-8') as f:
                sprint_time_key = json.loads(f.read()).get('sprint_time_key')
                # print(sprint_time_key)
                if sprint_time_key != None:
                    return sprint_time_key
                else:
                    return 0.9
    
    def __init_extra_sprint_time_key(self):

        # 确保存在文件夹
        if not os.path.exists('temp'):
            os.mkdir('temp')

        # 复制exe的images文件夹出来
        if os.path.exists('./temp/images'):
            # 如果目标文件夹已经存在，删除它
            # shutil.rmtree('./temp/images')
            pass
        else: 
            shutil.copytree(resource_path('images'), './temp/images')


        if not os.path.exists('./temp/settings.json'):
            with open('./temp/settings.json','w',encoding='utf-8') as f:
                f.write(json.dumps(dict(up_key='up',jump_key='ctrl',sprint_time_key=0.9,extra_sprint_time_key=0.0), ensure_ascii=False))
            return 0.0
        else:
            with open('./temp/settings.json','r',encoding='utf-8') as f:
                extra_sprint_time_key = json.loads(f.read()).get('extra_sprint_time_key')
                # print(extra_sprint_time_key)
                if extra_sprint_time_key != None:
                    return extra_sprint_time_key
                else:
                    return 0.0
    
    def __win(self):
        global VERSION
        self.title(f"自动跑「极速拼图」 {VERSION}")
        # 设置窗口大小、居中
        width = 420
        height = 230
        screenwidth = self.winfo_screenwidth()
        screenheight = self.winfo_screenheight()
        geometry = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2)
        self.geometry(geometry)
        self.resizable(width=False, height=False)

    def __tk_entry_up_key(self):

        entry = Entry(self)
        entry.insert(0, self.__init_up_key())
        entry.place(x=70, y=10, width=38)
        return entry
    
    def __tk_entry_jump_key(self):

        entry = Entry(self)
        entry.insert(0, self.__init_jump_key())
        entry.place(x=172, y=10, width=38)
        return entry
    
    def __tk_entry_sprint_time_key(self):

        entry = Entry(self)
        entry.insert(0, self.__init_sprint_time_key())
        entry.place(x=85, y=42, width=45)
        return entry
    
    def __tk_entry_extra_sprint_time_key(self):

        entry = Entry(self)
        entry.insert(0, self.__init_extra_sprint_time_key())
        entry.place(x=271, y=42, width=45)
        return entry

    def __tk_label_up_key(self):
        label = Label(self, text="向前键：")
        label.place(x=10, y=10)
        return label

    def __tk_label_sprint_time_key(self):
        label = Label(self, text="冲刺几秒：")
        label.place(x=10, y=42)
        return label
    
    def __tk_label_extra_sprint_time_key(self):
        label = Label(self, text="随机增加冲刺几秒：")
        label.place(x=135, y=42)
        return label
    
    def __tk_label_jump_key(self):
        label = Label(self, text="跳跃键：")
        label.place(x=112, y=10)
        return label
    
    def __tk_text_lb4no8xs(self):
        text = Text(self)
        text.tag_config('green', font=('黑体', 10, 'bold') ,foreground='green')
        text.tag_config('red', font=('黑体', 10, 'bold'), foreground='red')
        text.place(x=10, y=74, width=400, height=143)
        return text

    def __tk_button_start(self):
        btn = Button(self, text="开始")
        btn.place(x=217, y=6, width=60, height=30)
        return btn

    def __tk_button_stop(self):
        btn = Button(self, text="停止")
        btn.place(x=283, y=6, width=60, height=30)
        return btn

    def __tk_button_tutorial(self):
        btn = Button(self, text="教程")
        btn.place(x=350, y=6, width=60, height=30)
        return btn

class Win(WinGUI):
    flag = False
    global PROCESS_FLAG
    def __init__(self):
        super().__init__()
        self.__event_bind()
        self.tutorial_image = ImageTk.PhotoImage(PIL.Image.open(resource_path('images/core/请框住验证码的位置.jpg')))

    # 读取图像，解决imread不能读取中文路径的问题
    def cv_imread(self, filePath):
        # 核心就是下面这句，一般直接用这句就行，直接把图片转为mat数据
        cv_img=cv2.imdecode(np.fromfile(filePath,dtype=np.uint8),-1)
        # imdecode读取的是rgb，如果后续需要opencv处理的话，需要转换成bgr，转换后图片颜色会变化
        # cv_img=cv2.cvtColor(cv_img,cv2.COLOR_RGB2BGR)
        return cv_img
    
    def __tk_entry_up_key_constraint(self):

        global UP_KEY,JUMP_KEY,SPRINT_TIME_KEY,EXTRA_SPRINT_TIME_KEY
        # 确保存在文件夹
        if not os.path.exists('temp'):
            os.mkdir('temp')

        up_key = self.tk_entry_up_key.get()
        jump_key = self.tk_entry_jump_key.get()
        sprint_time_key = self.tk_entry_sprint_time_key.get()
        extra_sprint_time_key = self.tk_entry_extra_sprint_time_key.get()

        if up_key in pyautogui.KEYBOARD_KEYS:

            if jump_key in pyautogui.KEYBOARD_KEYS:
                
                
                try:
                    sprint_time_key = float(sprint_time_key)
                    if sprint_time_key > 0:
                        SPRINT_TIME_KEY.value = sprint_time_key
                except:
                    messagebox.showwarning('警告','冲刺时间必须大于0')
                    return False
                
                try:
                    extra_sprint_time_key = float(extra_sprint_time_key)
                    if extra_sprint_time_key > 0:
                        EXTRA_SPRINT_TIME_KEY.value = extra_sprint_time_key
                except:
                    messagebox.showwarning('警告','随机冲刺时间必须大于0')
                    return False

                UP_KEY.value = up_key
                JUMP_KEY.value = jump_key

                with open('./temp/settings.json','w',encoding='utf-8') as f:
                    f.write(json.dumps(dict(up_key=up_key,jump_key=jump_key,sprint_time_key=sprint_time_key,extra_sprint_time_key=extra_sprint_time_key), ensure_ascii=False))
        
                return True
    
            else:
                messagebox.showwarning('警告','不支持这个跳跃键')
                return False
        else:
            messagebox.showwarning('警告','不支持这个向前键')
            return False

        

    def __open_url(self, url):
        webbrowser.open_new(url)

    # 打开文件夹目录
    def __open_folder(self):
        path = os.getcwd()

        if path:
            # Windows系统下使用explorer打开目录，MacOS和Linux系统下使用open命令打开目录
            if os.name == 'nt':
                subprocess.run(['explorer', resource_path('images\\core',debug=True)])
            elif os.name == 'posix':
                subprocess.run(['open', resource_path('images/core',debug=True)])

    def __button_tutorial_window(self):
        # 创建一个新窗口
        new_window = Toplevel()

        # 设置窗口大小、居中
        width = 830
        height = 1250
        screenwidth = self.winfo_screenwidth()
        screenheight = self.winfo_screenheight()
        geometry = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2)
        new_window.geometry(geometry)
        new_window.resizable(width=False, height=False)

        ico_path = resource_path('icons/talesrunner.ico')
        new_window.iconbitmap(ico_path)

        # 程序前置条件说明
        tutorial_title_label = Label(new_window, text='启动程序前置条件', font=('黑体', 25, 'bold'), foreground='red', anchor='center', justify='center')
        tutorial_title_label.pack(pady=10)

        tutorial_title_label = Label(new_window, text='①管理员权限运行\n\n②已经进入了「极速拼图」游戏房间\n\n③「手动截图」替换「跑图素材图片」', font=('黑体', 18, 'bold'), anchor='center', justify='center')
        tutorial_title_label.pack(pady=10)

        images_button = Button(new_window, text='点击打开跑图图标目录, 修改成自己的图标', command=self.__open_folder, font=('黑体', 20, 'bold'), foreground='red', background="white")
        images_button.pack(pady=30)

        # 程序前置条件说明补充
        tutorial_title_two_label = Label(new_window, text='----------------本程序基于「彩色图像识别」----------------\n\n「跑图素材图片」的分辨率清晰度\n\n取决于「图像识别」的准确率', font=('黑体', 18, 'bold'), anchor='center', justify='center')
        tutorial_title_two_label.pack(pady=10)

        # 验证码状态说明
        tutorial_images_label = Label(new_window, image=self.tutorial_image, anchor='center', justify='center')
        tutorial_images_label.pack(pady=10)

        tutorial_title_two_label = Label(new_window, text='-------------------验证码状态说明-------------------\n\n这张图片需要「单独抠下来」里面的数字不重要\n\n只需要截图「任意验证码」即可', font=('黑体', 18, 'bold'), anchor='center', justify='center')
        tutorial_title_two_label.pack(pady=10)

        tutorial_title_two_label = Label(new_window, text='----------------如果出现无法识别的情况----------------\n\n①手动调整「游戏分辨率」(1280x960,1280x1024等等)\n\n②手动截图「新图标」替换「默认的图标」', font=('黑体', 18, 'bold'), foreground='red', anchor='center', justify='center')
        tutorial_title_two_label.pack(pady=10)

        tutorial_title_two_label = Label(new_window, text='-----------------------参数说明-----------------------\n\n冲刺时间「角色向前冲多久开始连跳」\n\n随机增加冲刺时间「避免卡住跳不过去」', font=('黑体', 18, 'bold'), anchor='center', justify='center')
        tutorial_title_two_label.pack(pady=10)

        # 项目信息
        frame = Frame(new_window)
        frame.pack(pady=10)
        frame_left = Frame(frame)
        frame_left.pack(side='left')
        frame_right = Frame(frame)
        frame_right.pack(side='right')

        github_label = Label(frame_left, text='GitHub仓库', font=('黑体', 15, 'bold','underline'), foreground='blue', anchor='w', underline=True)
        github_label.pack(padx=20)
        github_label.bind("<Button-1>", lambda event: self.__open_url("https://github.com/mochazi/AutoRunMap"))

        github_release_label = Label(frame_right, text='最新版本', font=('黑体', 15, 'bold','underline'), foreground='blue', anchor='w')
        github_release_label.pack(padx=20)
        github_release_label.bind("<Button-1>", lambda event: self.__open_url("https://github.com/mochazi/AutoRunMap/releases"))

    def __event_bind(self):
        self.tk_button_start.config(command=self.start)
        self.tk_button_stop.config(command=self.stop)
        self.tk_button_tutorial.config(command=self.__button_tutorial_window)

    def start(self):
        global PROCESS_WINDOWS_NAME,WINDOWS_NAME,PROCESS_FLAG,SPRINT_TIME_KEY,EXTRA_SPRINT_TIME_KEY,UP_KEY,JUMP_KEY
        if WINDOWS_NAME:
            self.thread_flag = True
            PROCESS_FLAG.value = True
            if self.__tk_entry_up_key_constraint():

                # 条件变量
                self.condition = threading.Condition()

                # 主线程（检测是否游戏准备、开始）
                self.main_thread = threading.Thread(target=self.print_info)
                self.main_thread.setDaemon(True) 
                self.main_thread.start()

                # 检测是否存在验证码
                self.click_verification_code_images_thread = threading.Thread(target=self.run_click_verification_code_images)
                self.click_verification_code_images_thread.setDaemon(True) 
                self.click_verification_code_images_thread.start()

                # 游戏控制输入（进程启动）
                self.press_process = multiprocessing.Process(target=Win.run_press, args=(PROCESS_WINDOWS_NAME,PROCESS_FLAG,SPRINT_TIME_KEY,EXTRA_SPRINT_TIME_KEY,UP_KEY,JUMP_KEY))
                self.press_process.daemon = True
                self.press_process.start()

                
                # 游戏控制输入（线程启动）
                # self.press_thread = threading.Thread(target=self.run_press)
                # self.press_thread.setDaemon(True) 
                # self.press_thread.start()
                
                

                
        else:
            messagebox.showwarning('警告','没有找到游戏窗口\n\n请使用窗口化而不是无边框')

    def stop(self):

        global PROCESS_FLAG
        self.thread_flag = False
        PROCESS_FLAG.value = False
        
        INFO = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S ') + '跑图任务已停止'
        self.tk_text_lb4no8xs.insert(END, "\n" + INFO + "\r\n", 'red')
        INFO = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S ') + '如果上一轮任务没结束, 将完成后再停止'
        self.tk_text_lb4no8xs.insert(END, "\n" + INFO + "\r\n\n", 'red')
        self.tk_text_lb4no8xs.see(END)
    

    # 检测是否游戏准备、开始
    def run_check_reday_or_start(self):


        try:
            pyautogui.press('esc')
            auto_run_map = AutoRunMap()
            work_list = [(resource_path(f'images/core/reday.jpg',debug=True), 'F9'), (resource_path(f'images/core/start.jpg',debug=True), 'F10')]

            with LOCK:
                # print("正在检查「游戏准备」「游戏开始」")
                INFO = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S ') + '正在检查「游戏准备」「游戏开始」'
                self.tk_text_lb4no8xs.insert(END, INFO + "\r\n", 'green')
                self.tk_text_lb4no8xs.see(END)

            for work in work_list:
                image_path = work[0]
                keys = work[1]
                
                avg = None
                if keys == 'F9':
                    avg,_ = auto_run_map.get_precise_xy(image_path,check_reday_or_start=True)
                if keys == 'F10':
                    avg,_ = auto_run_map.get_simple_xy(image_path)

                if avg:

                    if keys == 'F9':
                        # with LOCK:
                            # print("按下F9「游戏准备」")
                            # INFO = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S ') + '按下「游戏准备」'
                            # self.tk_text_lb4no8xs.insert(END, INFO + "\r\n", 'green')
                            # self.tk_text_lb4no8xs.see(END)
                        pyautogui.press(keys)
                        time.sleep(0.1)
                        pyautogui.press('F10') # 提高速度，不用再识别一次F10
                        return

                    if keys == 'F10':
                        # with LOCK:
                        #     print("按下F10「游戏开始」")
                        #     INFO = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S ') + '按下「游戏开始」'
                        #     self.tk_text_lb4no8xs.insert(END, INFO + "\r\n", 'green')
                        #     self.tk_text_lb4no8xs.see(END)
                        pyautogui.press(keys)
                          
        except:
            with LOCK:
                INFO = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S ') + '检查「游戏准备」「游戏开始」线程中止'
                logger.error(traceback.format_exc())
                self.tk_text_lb4no8xs.insert(END, INFO + "\r\n", 'red')
                self.tk_text_lb4no8xs.see(END)
                # self.stop()

    # 高精度sleep提高优先级，防止time.sleep调度了其他线程
    def high_precision_sleep(self,seconds):
        start_time = time.perf_counter()
        while time.perf_counter() - start_time < seconds:
            pass
    

    # 游戏控制输入
    # def run_press(self):

    #     try:
    #         # 置顶窗口，关闭多余窗口，开始跑图
    #         auto_run_map = AutoRunMap()
    #         auto_run_map.get_windows_location()

    #         pydirectinput.keyDown(self.up_key)

    #         while self.thread_flag:
    #             try:
    #                 self.high_precision_sleep(0.8926)

    #                 # 运动补偿，让冲刺久一点
    #                 # if random.choice([True,False]):
    #                 #     print( datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S ')+'延迟了')
    #                 #     self.high_precision_sleep(0.4)
                    
    #                 # 跳跃间隔
    #                 pydirectinput.press(self.jump_key)
    #                 self.high_precision_sleep(0.105)
    #                 pydirectinput.press(self.jump_key)
    #             except:
    #                 print(traceback.format_exc())
    #                 logger.error(traceback.format_exc())

    #         if self.thread_flag == False:
    #             pydirectinput.keyUp(self.up_key)
    #             auto_run_map.get_windows_location()

    #     except:
    #         with LOCK:
    #             INFO = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S ') + '游戏控制输入线程中止'
    #             logger.error(traceback.format_exc())
    #             self.tk_text_lb4no8xs.insert(END, INFO + "\r\n", 'red')
    #             self.tk_text_lb4no8xs.see(END)
    #             # self.stop()

    @staticmethod
    def run_press(PROCESS_WINDOWS_NAME,PROCESS_FLAG,sprint_time_key,extra_sprint_time_key,up_key, jump_key):
        
        if up_key.value not in pyautogui.KEYBOARD_KEYS or jump_key.value not in pyautogui.KEYBOARD_KEYS:
            error = f"向前键或跳跃键错误\nup_key: {up_key.value}\njump_key: {jump_key.value}\n子进程退出"
            # print(f'向前键或跳跃键错误')
            # print(f'up_key: {up_key.value}')
            # print(f'jump_key: {jump_key.value}')
            # print(f'游戏控制输入进程退出')
            logger.error(error)
            sys.exit()
        
        if sprint_time_key.value < 0 or extra_sprint_time_key.value < 0:
            error = f"冲刺时间或随机冲刺随机错误\nsprint_time_key: {sprint_time_key.value}\nextra_sprint_time_key: {extra_sprint_time_key.value}\n子进程退出"
            # print(f'向前键或跳跃键错误')
            # print(f'sprint_time_key: {sprint_time_key.value}')
            # print(f'extra_sprint_time_key: {extra_sprint_time_key.value}')
            # print(f'游戏控制输入进程退出')
            logger.error(error)
            sys.exit()
        
        
        def high_precision_sleep(seconds):
            start_time = time.perf_counter()
            while time.perf_counter() - start_time < seconds:
                pass
        
        def get_windows_location():

            try:
                time.sleep(0.2)
                app = pywinauto.Application().connect(title_re=PROCESS_WINDOWS_NAME.value)
                
                # 获取主窗口
                main_window = app.window(title=PROCESS_WINDOWS_NAME.value)

                # 将窗口置顶
                main_window.set_focus()
                main_window.topmost = True

                window = app.top_window()
                left, right, top, down  = window.rectangle().left, window.rectangle().right, window.rectangle().top, window.rectangle().bottom
                pyautogui.moveTo(left+300, top+300)
                # print(f"The window position is ({left}, {right}, {top}, {down})")
                return left, right, top, down
            except pywinauto.findwindows.ElementNotFoundError:
                print(traceback.format_exc())
                # INFO = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S ') + '没有匹配目标, 请确保游戏窗口没有脱离显示器屏幕'
                # logger.error(traceback.format_exc())
            except TypeError:
                print(traceback.format_exc())

        try:

            print("[游戏控制输入进程]启动")
            # 置顶窗口，关闭多余窗口，开始跑图
            get_windows_location()
            
            pydirectinput.keyDown(up_key.value)

            while PROCESS_FLAG.value:
                # print(PROCESS_FLAG.value)
                try:
                    # print( "冲刺开始" + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S '))
                    high_precision_sleep(sprint_time_key.value)
                    # print( "冲刺结束" + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S '))


                    # 运动补偿，让冲刺久一点
                    if extra_sprint_time_key.value:
                        if random.choice([True,False]):
                            # print( "随机冲刺开始" + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S '))
                            high_precision_sleep(extra_sprint_time_key.value)
                            # print( "随机冲刺结束" + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S '))

                    
                    # 跳跃间隔
                    pydirectinput.press(jump_key.value)
                    high_precision_sleep(0.105)
                    pydirectinput.press(jump_key.value)
                except:
                    print(traceback.format_exc())
                    logger.error(traceback.format_exc())

            if not PROCESS_FLAG.value:
                print(f'[游戏控制输入进程]已退出')
                pydirectinput.keyUp(up_key.value)
                get_windows_location()

        except:
            print(f'[游戏控制输入进程]错误')
            print(f'[游戏控制输入进程]已退出')
            print(traceback.format_exc())
            logger.error(f"[游戏控制输入进程]错误\n[游戏控制输入进程]已退出\n{traceback.format_exc()}")
            sys.exit()
 

    # 检测是否存在验证码
    def run_click_verification_code_images(self):

        print(f'[检查屏幕验证码线程]启动')
        while self.thread_flag:

            try:
                time.sleep(1)
                
                with LOCK:
                    # print("正在检查屏幕验证码")
                    INFO = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S ') + '正在检查屏幕验证码'
                    self.tk_text_lb4no8xs.insert(END, INFO + "\r\n", 'green')
                    self.tk_text_lb4no8xs.see(END)

                auto_run_map = AutoRunMap()
                auto_run_map.get_windows_location()

                
                # 裁剪验证码
                _,status = auto_run_map.cropped_verification_code_images()

                if status:
                    
                    # 识别验证码
                    # print("识别验证码开始" + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S '))
                    click_list = NumberOCR.ocr_working()
                    # print("识别验证码结束" + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S '))

                    if click_list:
                        with LOCK:
                            verification_code = ''.join([str(num) for num in click_list])
                            # print(f"验证码: {verification_code}")
                            INFO = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S ') + '验证码为' + f"「{verification_code}」"
                            self.tk_text_lb4no8xs.insert(END, INFO + "\r\n", 'green')
                            self.tk_text_lb4no8xs.see(END)
                            INFO = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S ') + "正在点击验证码"
                            self.tk_text_lb4no8xs.insert(END, INFO + "\r\n", 'green')
                            self.tk_text_lb4no8xs.see(END)

                        # 点击验证码
                        auto_run_map.simple_click_verification_code_images(click_list)
                        with LOCK:
                            INFO = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S ') + "验证码点击完毕"
                            self.tk_text_lb4no8xs.insert(END, INFO + "\r\n", 'green')
                            self.tk_text_lb4no8xs.see(END)
                    else:
                        pass
                        # print('识别验证码失败')
                else:
                    pass
                    # print('裁剪验证码失败')
                
                pyautogui.press('esc')

                # 唤醒主线程
                with self.condition:
                    self.condition.notify()

            except:
                with LOCK:
                    print(f'[检查屏幕验证码线程]错误')
                    print(traceback.format_exc())
                    INFO = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S ') + '识别验证码线程中止'
                    logger.error(traceback.format_exc())
                    self.tk_text_lb4no8xs.insert(END, INFO + "\r\n", 'red')
                    self.tk_text_lb4no8xs.see(END)
                    # self.stop()
        
        print(f'[检查屏幕验证码线程]结束')


    # 检测是否游戏准备、开始
    def print_info(self):

        global INFO,LOCATION

        auto_run_map = AutoRunMap()

        print(f'[检测是否「游戏准备」「开始」线程]启动')
        with self.condition:

            while self.thread_flag:
                
                try:
                    self.condition.wait()
                    self.run_check_reday_or_start()

                    # INFO = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + INFO
                    # self.tk_text_lb4no8xs.insert(1.0, INFO + "\r\n")
                
                except FileNotFoundError:
                    with LOCK:
                        logger.error(traceback.format_exc())
                        INFO = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S ') + '跑图图像文件不存在, 正在重新生成文件'
                        self.tk_text_lb4no8xs.insert(END, INFO + "\r\n", 'red')
                        self.tk_text_lb4no8xs.see(END)

                        # 复制exe的images文件夹出来
                        if os.path.exists('./temp/images'):
                            # 如果目标文件夹已经存在，删除它
                            shutil.rmtree('./temp/images')
                            shutil.copytree(resource_path('images'), './temp/images')

                        auto_run_map.get_windows_location()
                        time.sleep(1)
                        pyautogui.press('esc')
                        time.sleep(1)
                        pyautogui.press('esc')

                        INFO = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S ') + '跑图图像文件生成完毕, 即将开始跑图'
                        self.tk_text_lb4no8xs.insert(END, INFO + "\r\n", 'green')
                        self.tk_text_lb4no8xs.see(END)
                    
                except Exception as error:
                    with LOCK:
                        print(f'[检测是否「游戏准备」「开始」线程]错误')
                        print(traceback.format_exc())
                        INFO = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S ') + '没有匹配目标, 请确保游戏窗口没有脱离显示器屏幕'
                        logger.error(traceback.format_exc())
                        self.tk_text_lb4no8xs.insert(END, INFO + "\r\n", 'red')
                        self.tk_text_lb4no8xs.see(END)
                        # self.stop()

        print(f'[检测是否「游戏准备」「开始」线程]结束')

# 提升为管理员权限
def is_admin():
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False
    
if __name__ == "__main__":

    # On Windows calling this function is necessary.
    multiprocessing.freeze_support()

    # 进程共享字符串
    PROCESS_MANAGER = multiprocessing.Manager() 
    UP_KEY = PROCESS_MANAGER.Value(c_char_p, "up")
    JUMP_KEY = PROCESS_MANAGER.Value(c_char_p, "ctrl")
    SPRINT_TIME_KEY = PROCESS_MANAGER.Value("f", 0.9)
    EXTRA_SPRINT_TIME_KEY = PROCESS_MANAGER.Value("f", 0.0)
     
    # 检查游戏窗口是否存在（可能存在的几个名字）
    for name in ["Tales Runner", "Tales Runner ver."]:
        if check_window_exist(name):
            WINDOWS_NAME = name

    PROCESS_WINDOWS_NAME = PROCESS_MANAGER.Value(c_char_p, "Tales Runner")
    if WINDOWS_NAME:
        PROCESS_MANAGER.Value(c_char_p, WINDOWS_NAME)

    if is_admin():
        win = Win()
        ico_path = resource_path('icons/talesrunner.ico')
        win.iconbitmap(ico_path)
        win.mainloop()
    else:
        # Re-run the program with admin rights
        ctypes.windll.shell32.ShellExecuteW(None,"runas", sys.executable, '', None, 1)
        
    