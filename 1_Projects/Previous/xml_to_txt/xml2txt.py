import glob

import os

from xml.dom import minidom
 
classes = ["armor_1_red", "armor_2_red","armor_3_red","armor_4_red", "armor_5_red",
           "armor_1_blue","armor_2_blue","armor_3_blue","armor_4_blue","armor_5_blue",
           "car"]
 
def convert(size, box):
 
    dw = 1.0/size[0]
    dh = 1.0/size[1]
    x = (box[0]+box[1])/2.0
    y = (box[2]+box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    label = [x,y,w,h]
    return label
 
if __name__ == '__main__':
    all_files = glob.glob("./xmls/*.xml")
    for xml_path in all_files:
        dom = minidom.parse(xml_path)
       
        print(">" + xml_path[7:-3] + "txt")

        file_name = dom.getElementsByTagName("filename")[0].firstChild.data
        with open("./labels/" + str(xml_path)[7:-3] + "txt",'a') as f:
            objects = dom.getElementsByTagName("object") 
            size = dom.getElementsByTagName("size")[0]
            w = float(size.getElementsByTagName("width")[0].firstChild.data)
            h = float(size.getElementsByTagName("height")[0].firstChild.data)

            for i in range(len(objects)):
                name = objects[i].getElementsByTagName("name")[0]

                if name.firstChild.data == "armor":
                    armor_class = objects[i].getElementsByTagName("armor_class")[0]
                    armor_color = objects[i].getElementsByTagName("armor_color")[0]
                    bndbox = objects[i].getElementsByTagName("bndbox")[0]
                    if armor_class.firstChild.data != "none" and (armor_color.firstChild.data == "blue" or armor_color.firstChild.data == "red"):
                        if int(armor_class.firstChild.data) <= 5:
                            # print("object[" + str(i+1) + "]:" + 
                                # str(name.firstChild.data + "_" + armor_class.firstChild.data + "_" + armor_color.firstChild.data))
                            box = (float(bndbox.getElementsByTagName("xmin")[0].firstChild.data),
                                    float(bndbox.getElementsByTagName("xmax")[0].firstChild.data),
                                    float(bndbox.getElementsByTagName("ymin")[0].firstChild.data),
                                    float(bndbox.getElementsByTagName("ymax")[0].firstChild.data))
                            
                            label = convert((w,h),box)
                            
                            f.write(str(classes.index(name.firstChild.data + "_" + armor_class.firstChild.data + "_" + armor_color.firstChild.data))
                                    + " " + str(label[0]) + " " + str(label[1]) + " " + str(label[2]) + " " + str(label[3]) + "\n")

                if name.firstChild.data == "car":  
                    bndbox = objects[i].getElementsByTagName("bndbox")[0]

                    # print("object[" + str(i+1) + "]:" + str(name.firstChild.data))
                    box = (float(bndbox.getElementsByTagName("xmin")[0].firstChild.data),
                            float(bndbox.getElementsByTagName("xmax")[0].firstChild.data),
                            float(bndbox.getElementsByTagName("ymin")[0].firstChild.data),
                            float(bndbox.getElementsByTagName("ymax")[0].firstChild.data))
                   
                    label = convert((w,h),box)
                            
                    f.write(str(classes.index(name.firstChild.data)) + " " + str(label[0]) + " " + str(label[1]) + " " + str(label[2]) + " " + str(label[3]) + "\n")


            
