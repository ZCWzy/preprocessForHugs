from PIL import Image
import os
import cv2
import argparse

#segamentation反色，并且移动到文件夹4d-human/sam sagemention中

#png to jpg。这种情况下phalp才能正常工作


def main(opt):

    for picture in os.listdir(opt.images_dir):
        #png to jpg
        img = cv2.imread(opt.images_dir+"/"+picture)
        cv2.imwrite(os.path.join(opt.phalp_output_dir+'/'+picture+'.jpg'), img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    for picture in os.listdir(opt.seg_dir):
        # 反转颜色
        im = Image.open(opt.seg_dir + "/" + picture)
        im_inverted = im.point(lambda _: 255-_)
        im_inverted.save(os.path.join(opt.seg_output_dir+"/"+picture))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', type=str, default=None, required=True)
    parser.add_argument('--seg_dir', type=str, default=None, required=True)
    parser.add_argument('--seg_output_dir', type=str, default=None, required=True)
    parser.add_argument('--phalp_output_dir', type=str, default=None, required=True)
    opt = parser.parse_args()
    main(opt)
