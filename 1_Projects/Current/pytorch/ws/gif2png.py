from PIL import Image, ImageSequence
import glob
image_path = glob.glob(
    '/workspaces/workspace/1_Projects/Current/pytorch/ws/data/test/1st_manual/*.gif')


for i in image_path:
    im = Image.open(i)  # 使用Image的open函数打开test.gif图像
    name = [i.split('/')[-1].split('.')[0] if i.split('/')
            [-1].split('.')[1] == 'gif' else ''][0]
    print(name)
    index = 1
    for frame in ImageSequence.Iterator(im):  # for循环迭代的取im里的帧
        # 取到一个帧调用一下save函数保存
        frame.save(
            "/workspaces/workspace/1_Projects/Current/pytorch/ws/data/test/1st_manual/"+str(name)+".png")
        index += 1  # 序号依次叠加
