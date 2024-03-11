import tkinter as tk
from tkinter import filedialog


def getFilePath():
    # 实例化
    root = tk.Tk()
    root.withdraw()

    # 获取文件夹路径
    gifFileName = filedialog.askopenfilename()
    
    return gifFileName
