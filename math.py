


def img_read(filename):
    return cv2.imdecode(np.fromfile(filename, dtype=np.uint8), cv2.IMREAD_COLOR)
    # 以uint8方式读取filename 放入imdecode中，cv2.IMREAD_COLOR读取彩色照片