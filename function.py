






def img_first_pre(self, car_pic_file):
    """
    :param car_pic_file: 图像文件
    :return:已经处理好的图像文件 原图像文件
    """
    if type(car_pic_file) == type(""):
        img = img_math.img_read(car_pic_file)
    else:
        img = car_pic_file

    pic_hight, pic_width = img.shape[:2]
    if pic_width > MAX_WIDTH:
        resize_rate = MAX_WIDTH / pic_width
        img = cv2.resize(img, (MAX_WIDTH, int(pic_hight * resize_rate)), interpolation=cv2.INTER_AREA)
    # 缩小图片

    blur = 5
    img = cv2.GaussianBlur(img, (blur, blur), 0)
    oldimg = img
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 转化成灰度图像

    Matrix = np.ones((20, 20), np.uint8)
    img_opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, Matrix)
    img_opening = cv2.addWeighted(img, 1, img_opening, -1, 0)
    # 创建20*20的元素为1的矩阵 开操作，并和img重合

    ret, img_thresh = cv2.threshold(img_opening, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img_edge = cv2.Canny(img_thresh, 100, 200)
    # Otsu’s二值化 找到图像边缘

    Matrix = np.ones((4, 19), np.uint8)
    img_edge1 = cv2.morphologyEx(img_edge, cv2.MORPH_CLOSE, Matrix)
    img_edge2 = cv2.morphologyEx(img_edge1, cv2.MORPH_OPEN, Matrix)
    return img_edge2, oldimg
