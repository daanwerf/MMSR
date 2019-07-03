from __future__ import print_function
from __future__ import division

import cv2
from pywt import wavedec
import numpy as np


class Image:

    def __init__(self, category, iid):
        self.category = str(category)
        self.iid = str(iid + category*100)

        img_path = r"Corel100" + "\\" + self.category + "_" + self.iid \
                   + ".jpg"
        self.rgb_img = cv2.imread(img_path , 1)
        self.ycrcb_img = cv2.cvtColor(self.rgb_img, cv2.COLOR_BGR2YCR_CB)

    def get_image_id(self):
        return self.iid

    def get_file_name(self):
        return str(self.category) + "_" + str(self.iid)

    def get_image_category(self):
        return self.category

    def get_image_rgb(self):
        return self.rgb_img

    def show_image_rgb(self):
        cv2.imshow('image', self.rgb_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def show_image_ycbcr(self):
        cv2.imshow('image', self.ycrcb_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def get_Y_channel(self):
        s = cv2.split(self.ycrcb_img)
        return s[0]

    def show_Y_image(self):
        cv2.imshow('image', self.get_Y_channel())
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def get_cb_channel(self):
        s = cv2.split(self.ycrcb_img)
        return s[1]

    def show_cb_image(self):
        cv2.imshow('image', self.get_cb_channel())
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def get_cr_channel(self):
        s = cv2.split(self.ycrcb_img)
        return s[2]

    def show_cr_image(self):
        cv2.imshow('image', self.get_cr_channel())
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def get_canny_edge(self):
        sigma = 0.33
        v = np.median(self.get_Y_channel())
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        return cv2.Canny(self.get_Y_channel(), lower, upper)

    def show_canny_edge(self):
        cv2.imshow('image', self.get_canny_edge())
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def get_combined_Y_cb_cr(self):
        return cv2.merge((self.get_canny_edge(), self.get_cb_channel(), self.get_cr_channel()))

    def show_combined(self):
        cv2.imshow('image', self.get_combined_Y_cb_cr())
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def get_changed_rgb(self):
        return cv2.cvtColor(self.get_combined_Y_cb_cr(), cv2.COLOR_YCrCb2BGR)

    def show_changed_rgb(self):
        cv2.imshow('image', self.get_changed_rgb())
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def get_changed_b_channel(self):
        b,g,r = cv2.split(self.get_changed_rgb())
        return b

    def show_changed_b_channel(self):
        cv2.imshow('image', self.get_changed_b_channel())
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def get_changed_g_channel(self):
        b,g,r = cv2.split(self.get_changed_rgb())
        return g

    def show_changed_g_channel(self):
        cv2.imshow('image', self.get_changed_g_channel())
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def get_changed_r_channel(self):
        b,g,r = cv2.split(self.get_changed_rgb())
        return r

    def show_changed_r_channel(self):
        cv2.imshow('image', self.get_changed_r_channel())
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def compute_b_histogram(self):
        histSize = 256
        histRange = (0, 256)

        return cv2.calcHist(cv2.split(self.get_changed_rgb()), [0], None, [histSize], histRange, accumulate=False)

    def compute_g_histogram(self):
        histSize = 256
        histRange = (0, 256)

        return cv2.calcHist(cv2.split(self.get_changed_rgb()), [1], None, [histSize], histRange, accumulate=False)

    def compute_r_histogram(self):
        histSize = 256
        histRange = (0, 256)

        return cv2.calcHist(cv2.split(self.get_changed_rgb()), [2], None, [histSize], histRange, accumulate=False)

    def show_b_histogram(self):
        histSize = 256
        hist_w = 512
        hist_h = 400
        bin_w = int(round(hist_w / histSize))
        histImage = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)
        cv2.normalize(self.compute_b_histogram(), self.compute_b_histogram(), alpha=0, beta=hist_h,
                      norm_type=cv2.NORM_MINMAX)
        for i in range(1, histSize):
            cv2.line(histImage, (bin_w * (i - 1), hist_h - int(np.round(self.compute_b_histogram()[i - 1]))),
                    (bin_w * (i), hist_h - int(np.round(self.compute_b_histogram()[i]))),
                    (255, 0, 0), thickness=2)
        cv2.imshow('B channel histogram', histImage)
        cv2.waitKey()

    def show_g_histogram(self):
        histSize = 256
        hist_w = 512
        hist_h = 400
        bin_w = int(round(hist_w / histSize))
        histImage = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)
        cv2.normalize(self.compute_g_histogram(), self.compute_g_histogram(), alpha=0, beta=hist_h,
                      norm_type=cv2.NORM_MINMAX)
        for i in range(1, histSize):
            cv2.line(histImage, (bin_w * (i - 1), hist_h - int(np.round(self.compute_g_histogram()[i - 1]))),
                     (bin_w * (i), hist_h - int(np.round(self.compute_g_histogram()[i]))),
                     (0, 255, 0), thickness=2)
        cv2.imshow('G channel histogram', histImage)
        cv2.waitKey()

    def show_r_histogram(self):
        histSize = 256
        hist_w = 512
        hist_h = 400
        bin_w = int(round(hist_w / histSize))
        histImage = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)
        cv2.normalize(self.compute_r_histogram(), self.compute_r_histogram(), alpha=0, beta=hist_h, norm_type=cv2.NORM_MINMAX)
        for i in range(1, histSize):
            cv2.line(histImage, (bin_w * (i - 1), hist_h - int(np.round(self.compute_r_histogram()[i - 1]))),
                    (bin_w * (i), hist_h - int(np.round(self.compute_r_histogram()[i]))),
                    (0, 0, 255), thickness=2)
        cv2.imshow('R channel histogram', histImage)
        cv2.waitKey()

    def show_total_histogram(self):
        histSize = 256
        hist_w = 512
        hist_h = 400
        bin_w = int(round(hist_w / histSize))
        histImage = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)
        cv2.normalize(self.compute_b_histogram(), self.compute_b_histogram(), alpha=0, beta=hist_h,
                      norm_type=cv2.NORM_MINMAX)
        cv2.normalize(self.compute_g_histogram(), self.compute_g_histogram(), alpha=0, beta=hist_h,
                      norm_type=cv2.NORM_MINMAX)
        cv2.normalize(self.compute_r_histogram(), self.compute_r_histogram(), alpha=0, beta=hist_h,
                      norm_type=cv2.NORM_MINMAX)
        for i in range(1, histSize):
            cv2.line(histImage, (bin_w * (i - 1), hist_h - int(np.round(self.compute_b_histogram()[i - 1]))),
                    (bin_w * (i), hist_h - int(np.round(self.compute_b_histogram()[i]))),
                    (255, 0, 0), thickness=2)
            cv2.line(histImage, (bin_w * (i - 1), hist_h - int(np.round(self.compute_g_histogram()[i - 1]))),
                    (bin_w * (i), hist_h - int(np.round(self.compute_g_histogram()[i]))),
                    (0, 255, 0), thickness=2)
            cv2.line(histImage, (bin_w * (i - 1), hist_h - int(np.round(self.compute_r_histogram()[i - 1]))),
                    (bin_w * (i), hist_h - int(np.round(self.compute_r_histogram()[i]))),
                    (0, 0, 255), thickness=2)
        cv2.imshow('RGB histogram', histImage)
        cv2.waitKey()

    def get_b_hist_dwt(self):
        cA3, cD2, cD1, cD3 = wavedec(self.compute_b_histogram(), wavelet='haar', level=3)
        return cA3

    def get_g_hist_dwt(self):
        cA3, cD2, cD1, cD3 = wavedec(self.compute_g_histogram(), wavelet='haar', level=3)
        return cA3

    def get_r_hist_dwt(self):
        cA2, cD2, cD1 = wavedec(self.compute_r_histogram(), wavelet='haar', level=2)
        return cA2

    def get_feature_vector(self):
        return np.concatenate((self.get_b_hist_dwt(), self.get_g_hist_dwt(), self.get_r_hist_dwt()))

    def save_feature_vector(self):
        np.save('featuredb/' + self.category + "_" + self.iid, self.get_feature_vector())
