#coding=utf-8
import time,os,cv2,pyautogui,pywinauto,ddddocr,threading,pydirectinput,traceback
import numpy as np
from concurrent.futures import ProcessPoolExecutor

INFO = ''
LOCATION = None
lock = threading.Lock()


class NumberOCR():

    def __init__(self):
        self.ocr = ddddocr.DdddOcr(show_ad=False)
        # 阈值
        self.lowerb = [100,110,120,95]

    # 读取图像，解决imread不能读取中文路径的问题
    def cv_imread(self, filePath):
        # 核心就是下面这句，一般直接用这句就行，直接把图片转为mat数据
        cv_img=cv2.imdecode(np.fromfile(filePath,dtype=np.uint8),-1)
        # imdecode读取的是rgb，如果后续需要opencv处理的话，需要转换成bgr，转换后图片颜色会变化
        # cv_img=cv2.cvtColor(cv_img,cv2.COLOR_RGB2BGR)
        return cv_img
    
    # 拆分数字
    def split_number(self,num):
        num_str = str(num)
        digits = [int(digit) for digit in num_str]
        return digits

    # 判断是不是数字
    def is_number(self,str):
        return str.isdigit()

    # 验证码图片（预处理）
    def clear_images(self,image_path,lowerb):

        img = self.cv_imread(image_path)
        img2 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img2 = cv2.inRange(img2, lowerb=lowerb, upperb=255)
        # ret, img2 = cv2.threshold(img2, 120, 255, cv2.THRESH_BINARY)
        img2 = self.operate_img(img2, 1)
        cv2.imencode('.jpg', img2)[1].tofile(f'./temp/验证码降噪.jpg')


    # 识别验证码
    def ocr_working(self,image_path):

        for number in self.lowerb:
            self.clear_images(image_path,number)

            with open(f'./temp/验证码降噪.jpg', 'rb') as f:
                image = f.read()

            res = self.ocr.classification(image)

            if self.is_number(res):
                return self.split_number(res)
        
        return False

    # 计算邻域非白色个数
    def calculate_noise_count(self,img_obj, w, h):
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
    def operate_img(self,img,k):
        w,h = img.shape
        # 从高度开始遍历
        for _w in range(w):
            # 遍历宽度
            for _h in range(h):
                if _h != 0 and _w != 0 and _w < w-1 and _h < h-1:
                    if self.calculate_noise_count(img, _w, _h) < k:
                        img.itemset((_w,_h),255)
                        img.itemset((_w, _h), 255)
                        img.itemset((_w, _h), 255)
        return img

# 检测是否游戏准备、开始
def run_check_reday_or_start(image_path, keys):

    try:
        auto_run_map = AutoRunMap()
        auto_run_map.check_reday_or_start(image_path,keys)
        pyautogui.press('esc')
    except:
        print(traceback.format_exc())

# 游戏控制输入
def run_press():

    
    pydirectinput.keyDown('i')

    while True:
        try:
            pydirectinput.press('s')
            time.sleep(0.2)
        except:
            print(traceback.format_exc())

# 检测是否存在验证码
def run_click_verification_code_images():

    while True:

        try:
            auto_run_map = AutoRunMap()

            # 裁剪验证码
            _,status = auto_run_map.cropped_verification_code_images()

            if status:
                # 识别验证码
                click_list = auto_run_map.ocr.ocr_working('temp/验证码.jpg')
                print(f"验证码: {click_list}")
                if click_list:
                    # 点击验证码
                    auto_run_map.simple_click_verification_code_images(click_list)
                else:
                    print('识别验证码失败')
            else:
                print('裁剪验证码失败')
            
            time.sleep(1)
            pyautogui.press('esc')
        except:
            print(traceback.format_exc())

class AutoRunMap():

    def __init__(self):
        self.ocr = NumberOCR()

    # 读取图像，解决imread不能读取中文路径的问题
    def cv_imread(self,filePath):
        
        # 核心就是下面这句，一般直接用这句就行，直接把图片转为mat数据
        cv_img=cv2.imdecode(np.fromfile(filePath,dtype=np.uint8),-1)
        # imdecode读取的是rgb，如果后续需要opencv处理的话，需要转换成bgr，转换后图片颜色会变化
        # cv_img=cv2.cvtColor(cv_img,cv2.COLOR_RGB2BGR)
        return cv_img

    def get_windows_location(self):
        app = pywinauto.Application().connect(title_re="Tales Runner")
        
        # 获取主窗口
        main_window = app.window(title="Tales Runner")

        # 将窗口置顶
        main_window.set_focus()
        main_window.topmost = True

        window = app.top_window()
        left, right, top, down  = window.rectangle().left, window.rectangle().right, window.rectangle().top, window.rectangle().bottom
        # print(f"The window position is ({left}, {right}, {top}, {down})")
        return left, right, top, down

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
    def get_simple_xy(self, img_model_path=None,verification_code=False,name=None, show=False, output=True):
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
    
    # 裁剪图片
    def cropped_images(self, img_rgb):

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

        # 开始裁剪
        # cropped = cropped[cropped_Y:cropped_Y+cropHeight, cropped_X:cropped_X+cropWidth]

        cv2.imwrite("./temp/screenshot_cropped.jpg", cropped)
        img_rgb = cropped

        # 返回裁剪图片，x偏移量，y偏移量
        return img_rgb, cropped_X + left, cropped_Y + top


    # 自动点击
    def auto_click(self, var_avg):
        """
        输入一个元组，自动点击
        :param var_avg: 坐标元组
        :return: None
        """
        self.get_windows_location()
        pyautogui.click(var_avg[0], var_avg[1], button='left')
        time.sleep(1)


    # 点击验证码（简单匹配）
    def simple_click_verification_code_images(self, click_list, show=False, output=True):

        global INFO

        # 确保存在文件夹
        if not os.path.exists('temp'):
            os.mkdir('temp')

        img = None
        # 排序（确定左和右）
        for index in click_list:

            fish_image_path = f'images/core/{index}.jpg'
            avg,img = self.get_simple_xy(fish_image_path)

            # INFO = f"正在点击图片{fish_image_path}"
            # print(INFO)

            self.auto_click(avg)
            pyautogui.moveTo(avg[0]+300, avg[1])


        # 展示图片
        if show:
            cv2.imshow('img', img)
            cv2.waitKey()
        
        # 输出图片
        if output:
            cv2.imwrite('./temp/click_fish_images.jpg', img)

        if click_list == {}:
            return False
        else:
            INFO = f"点击鱼完毕"
             # print(INFO)
            return True
       
        # -------------------------------------------

    # 裁剪图片
    def cropped_verification_code_images(self,show=False, output=True):

        # 获取验证码坐标
        avg,img_rgb = self.get_simple_xy(f'images/core/请框住验证码的位置.jpg',verification_code=True)
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

        avg,_ = self.get_simple_xy(image_path)
        
        if avg:
            pyautogui.press(keys)
        

    def auto_run_map(self):

        global LOCATION
         

        while True:

            try:
                self.get_windows_location()
                time.sleep(0.1)
                with ProcessPoolExecutor(max_workers=2) as p:
                    work_list = [(f'images/core/start.jpg', 'F10'), (f'images/core/reday.jpg', 'F9')]
                    for work in work_list:
                        p.submit(run_check_reday_or_start,work[0], work[1])
            except:
                print(traceback.format_exc())


if __name__ == '__main__':

    auto_run_map = AutoRunMap()
    left, right, top, down = auto_run_map.get_windows_location()
    pyautogui.moveTo(left+200, top+200)


    # 游戏控制输入
    run_press_thread = threading.Thread(target=run_press)
    run_press_thread.daemon = True # 将线程设置为分离状态
    run_press_thread.start() # 启动线程

    # 检测是否存在验证码
    run_click_verification_code_images_thread = threading.Thread(target=run_click_verification_code_images)
    run_click_verification_code_images_thread.daemon = True # 将线程设置为分离状态
    run_click_verification_code_images_thread.start() # 启动线程

    
    (
        AutoRunMap().auto_run_map()
    )