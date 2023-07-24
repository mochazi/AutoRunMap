import cv2,pyautogui,time,pywinauto

def get_windows_location():
    app = pywinauto.Application().connect(title_re="Tales Runner")
    
    # 获取主窗口
    main_window = app.window(title="Tales Runner")

    # 将窗口置顶
    main_window.set_focus()
    main_window.topmost = True

    window = app.top_window()
    left, right, top, down  = window.rectangle().left, window.rectangle().right, window.rectangle().top, window.rectangle().bottom
    print(f"The window position is ({left}, {right}, {top}, {down})")
    return left, right, top, down

# 自动点击
def auto_click(var_avg):
    """
    输入一个元组，自动点击
    :param var_avg: 坐标元组
    :return: None
    """
    pyautogui.click(var_avg[0], var_avg[1], button='left')
    time.sleep(1)

def get_keypoint_bounds(kp):
    # 获取关键点的中心坐标和直径大小
    x, y = kp.pt
    diameter = kp.size

    # 计算关键点的左上角和右下角坐标
    x1 = int(x - diameter/2)
    y1 = int(y - diameter/2)
    x2 = int(x + diameter/2)
    y2 = int(y + diameter/2)

    return x1, y1, x2, y2

# 获取指定窗口坐标并置顶窗口
left, right, top, down = get_windows_location()

pyautogui.screenshot().save("./temp/screenshot.png")

# 读取原始图像和模板图像
img_rgb = cv2.imread('temp/screenshot.png')
template = cv2.imread('images/core/4.png')

# ---------------裁剪图片------------------

print(img_rgb.shape)

# cropped = img_rgb[top+300:down-800,left+400:right-300]  # 裁剪坐标为[y0:y1, x0:x1]
cropped = img_rgb[top:down,left:right]  # 裁剪坐标为[y0:y1, x0:x1]

height, width = cropped.shape[:2]

print((height, width))
# 计算中间区域的左上角X和Y坐标以及其宽度和高度
cropWidth = int(width * 0.55)
cropHeight = int(height * 0.9)
startX = (width - cropWidth) // 2
startY = (height - cropHeight) // 2

print()
print((cropWidth, cropHeight))
print((startX, startY))

cropped = cropped[startY:startY+cropHeight, startX:startX+cropWidth]

cv2.imwrite("./temp/screenshot_cropped.png", cropped)
img_rgb = cropped

#-----------------------------------------

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
for number in [0.3, 0.4, 0.5,0.6,0.7]:
    for m, n in matches:
        if m.distance < number * n.distance:
            good_matches.append(m)

# 绘制匹配结果
img_matches = cv2.drawMatches(template, kp1, img_rgb, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


# 获取所有好的匹配点在原始图像中的中心坐标
sum_x = 0
sum_y = 0
for match in good_matches:
    # 获取关键点在原始图像中的索引
    index = match.trainIdx

    # 获取关键点的左上角和右下角坐标
    x1, y1, x2, y2 = get_keypoint_bounds(kp2[index])

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

# 计算平均中心坐标
avg_x = int(sum_x / len(good_matches))
avg_y = int(sum_y / len(good_matches))

# 输出结果
x_click = avg_x + startX + left
y_click = avg_y + startY + top
print('平均中心坐标：({},{})'.format(x_click, y_click))

auto_click((x_click,y_click))

# # 显示结果
cv2.imshow('Matches', img_matches)
cv2.waitKey()
cv2.destroyAllWindows()

