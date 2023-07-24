
# 超级跑跑自动跑图

---

> **基于openCV的模板匹配、pyautogui模拟点击实现**
> 
> **缺点：python的openCV打包成exe占用过大**
>> **该项目仅限学习交流，请勿用于商业用途，如有侵权，请联系删除。**

---

# 环境

---

|**运行环境**|**项目使用版本**|
|:----:|:--------:|
|**python**|**3.7.9**|

---

# 构建

---

## 创建虚拟环境
```shell
python -m venv venv
```

## 进入虚拟环境
```shell
cd venv/Scripts/
```

## 激活虚拟环境
```
./activate
```

## 安装依赖
```shell
pip install -r requirements.txt
```

## 运行
```shell
python main.py
```

---

# 打包

```shell
pyinstaller --onefile -w -i "icons/talesrunner.ico" --add-data "images/core/*;images/core/" --add-data "icons/*;icons" --add-data "model/*;model" --upx-dir upx-4.0.2-win64 main.py
```