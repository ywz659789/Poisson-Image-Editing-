import cv2
import numpy as np
import Possion

def seamless_cloning(pic_index):
    mask = cv2.imread("../data/01/mask_0{0}.jpg".format(pic_index))
    src = cv2.imread("../data/01/source_0{0}.jpg".format(pic_index))
    dst = cv2.imread("../data/01/target_0{0}.jpg".format(pic_index))   
    mask = mask[:,:,0]

    result_0 = Possion.Seamless_cloning(src,dst,mask,0)   # 0；普通融合 
    result_1 = Possion.Seamless_cloning(src,dst,mask,1)   # 1.混合融合 
    cv2.imwrite("../result/01/result_0{0}.jpg".format(pic_index), result_0)
    cv2.imwrite("../result/01/result_0{0}_mixed.jpg".format(pic_index), result_1)  

def monochrome_transfer(pic_index):
    mask = cv2.imread("../data/02/mask_0{0}.jpg".format(pic_index))
    src = cv2.imread("../data/02/source_0{0}.jpg".format(pic_index))
    dst = cv2.imread("../data/02/target_0{0}.jpg".format(pic_index))  
    mask = mask[:,:,0]
    result= Possion.Seamless_cloning(src,dst,mask,2)   # 2.单色迁移
    cv2.imwrite("../result/02/result_0{0}.jpg".format(pic_index), result)

def texture_flattening(pic_index):
    mask = cv2.imread("../data/03/mask_0{0}.jpg".format(pic_index))
    src = cv2.imread("../data/03/source_0{0}.jpg".format(pic_index))
    mask = mask[:,:,0]
    result = Possion.Texture_flattening(src,mask,10,30)   
    cv2.imwrite("../result/03/result_0{0}.jpg".format(pic_index), result)

def local_illumination_changes(pic_index):
    mask = cv2.imread("../data/04/mask_0{0}.jpg".format(pic_index))
    src = cv2.imread("../data/04/source_0{0}.jpg".format(pic_index))
    mask = mask[:,:,0]
    result = Possion.Local_illumination_changes(src.astype(np.float64),mask,0.2,0.2)   
    cv2.imwrite("../result/04/result_0{0}.jpg".format(pic_index), result)

def local_color_changes(pic_index):
    mask = cv2.imread("../data/05/mask_0{0}.jpg".format(pic_index))
    src = cv2.imread("../data/05/source_0{0}.jpg".format(pic_index))
    mask = mask[:,:,0]
    result = Possion.Local_color_changes(src,mask,1.5,0.5,0.5)   
    cv2.imwrite("../result/05/result_0{0}.jpg".format(pic_index), result)

def seamless_tiling(pic_index):
    src = cv2.imread("../data/06/source_0{0}.jpg".format(pic_index))
    result = Possion.Seamless_tiling(src.astype(np.float64))   
    cv2.imwrite("../result/06/result_0{0}.jpg".format(pic_index), result)

def menu():
    print("请选择一个操作:")
    print("1. 无缝融合")
    print("2. 单色迁移")
    print("3. 纹理平滑")
    print("4. 局部照明变化")
    print("5. 局部颜色变化")
    print("6. 无缝拼接")
    choice = input("输入选项编号: ")

    if choice == '1':
        for pic_index in range(1, 9):
            seamless_cloning(pic_index)
    elif choice == '2':
        for pic_index in range(1, 2):
            monochrome_transfer(pic_index)
    elif choice == '3':
        for pic_index in range(1, 2):
            texture_flattening(pic_index)
    elif choice == '4':
        for pic_index in range(2, 3):
            local_illumination_changes(pic_index)
    elif choice == '5':
        for pic_index in range(1, 2):
            local_color_changes(pic_index)
    elif choice == '6':
        for pic_index in range(1, 4):
            seamless_tiling(pic_index)
    else:
        print("无效的选择，请重新运行程序并选择有效的编号。")
    
if __name__ == "__main__":
    menu()




    


