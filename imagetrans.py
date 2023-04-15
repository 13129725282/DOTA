# 实现从png到jpg

import dota_utils as util
import cv2
import os

# this code is used to convert image formats
# from PNG to JPG
def imageformatTrans(srcpath, dstpath, format):
    filelist = util.GetFileFromThisRootDir(srcpath)
    for fullname in filelist:
        img = cv2.imread(fullname)
        basename = util.custombasename(fullname)
        dstname = os.path.join(dstpath, basename + format)
        cv2.imwrite(dstname, img)

if __name__ == '__main__':
    # an example

    imageformatTrans(r'G:\DOTA\train\other_object\aug\total\images', r'G:\DOTA\train\other_object\aug\total\images_yolo',
                     '.jpg')