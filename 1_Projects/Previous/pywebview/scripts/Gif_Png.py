from PIL import Image
import os


def gif2png(fileImportPath,fileExportPath):
   
    try:
        # 获取文件夹路径
        gifFileName = fileImportPath

        #使用Image模块的open()方法打开gif动态图像时，默认是第一帧
        im = Image.open(gifFileName)
        imread = Image.open(gifFileName)
        
        pngDir = gifFileName[:-4]
        if len(os.listdir(pngDir)) != 0 or os.path.exists(pngDir):
            return -1
        
        #创建存放每帧图片的文件夹
        if os.path.exists(fileExportPath) and not os.path.isfile(fileExportPath) :
            pngDir = fileExportPath
            pass
        elif fileExportPath:
            pngDir = fileExportPath
            os.mkdir(pngDir) 
        else:
            os.mkdir(pngDir)      
            
        count = 1
        print("---------")
    except Exception as e:
        print(e)
        return -1
    try:
        try:
            while True:
                print("正在读取["+str(count)+"]")
                current = im.tell()
                
                #获取下一帧图片
                im.seek(current+1)
                count += 1
        except EOFError:
            pass
    except Exception as e:
        print(e)
        return -1
    print("---------")
    print("读取完成,共"+str(count)+"张照片")
    print("---------")
    try:
        try:
            while True:
                #保存当前帧图片
                current = imread.tell()
                imread.save(pngDir+'/'+str(current)+'.png')
                #获取下一帧图片
                imread.seek(current+1)
                print("正在保存第"+str(current+1)+"/"+str(count)+"张图片")
        except EOFError:
                pass
    except Exception as e:
        print(e)
        return -1
    print("保存完成")
    print("---------")
    
    return 1
    
    
    
def png2gif():
    pass