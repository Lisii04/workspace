import random
import sys
import threading
import time
import webview




class Api:

    def __init__(self):
        pass
    def getImportFilePath(self):
        from scripts.getFilePath import getFilePath
        return getFilePath()
    
    def gifToPng(self,fileImportPath,fileExportPath):
        from scripts.Gif_Png import gif2png
        return gif2png(fileImportPath,fileExportPath)

    def sayHelloTo(self, name):
        response = {'message': 'Hello {0}!'.format(name)}
        return response
    


if __name__ == '__main__':
    api = Api()
    window = webview.create_window('EazyTrans', 
                                   "index/main.html", 
                                   js_api=api,
                                   min_size=(1100,700)
                                   )
    webview.start(debug=False)