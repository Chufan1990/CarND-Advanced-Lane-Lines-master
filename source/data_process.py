import numpy as np
import cv2
import glob
import function_definition as fd


def data_process(img, mtx, dist):
    undist = cv2.undistort(img, mtx, dist, mtx)
    warped, M, Minv = fd.img_warp(undist)
    sobelx = fd.abs_sobel_thresh(warped, orient='x', thresh=(30, 150))
    sobely = fd.abs_sobel_thresh(warped, orient='y', thresh=(30, 150))
    mag_binary = fd.mag_thresh(warped, sobel_kernel=3, mag_thresh=(10, 200))
    dir_binary = fd.dir_threshold(warped, sobel_kernel=3, thresh=(0.6, 1.3))
    # hls_binary = fd.hls_select(warped, thresh=(100, 255))
    combine = np.zeros_like(sobelx)
    combine[((sobelx == 1) | (sobely == 1)) | (
        (mag_binary == 1) & (dir_binary == 1))] = 1
        # (mag_binary == 1) & (dir_binary == 1)) | (hls_binary == 1)] = 1
    # left_fit, right_fit, left_curverad, right_curverad=fd.lane_find(combine)
    left_fit, right_fit =fd.lane_find(combine)
    xsize=warped.shape[1]
    ysize=warped.shape[0]
    region_select=np.copy(warped)
    XX, YY=np.meshgrid(np.arange(0, xsize), np.arange(0, ysize))
    region_thresholds=(XX > (left_fit[0] * YY**2 + left_fit[1] * YY + left_fit[2])) & (
        XX < (right_fit[0] * YY**2 + right_fit[1] * YY + right_fit[2]))
    print("\n region_thresholds \n",region_thresholds)
    print("\n region_selection \n",region_select)
#     region_select[region_thresholds]=[0, 180, 0]
#     origin_img=cv2.warpPerspective(
#         region_select, Minv, (region_select.shape[1], region_select.shape[0]))
#     result=cv2.addWeighted(undist, 1, origin_img, 0.5, 0)
# #    font=cv2.InitFont(cv2.CV_FONT_HERSHEY_SCRIPT_SIMPLEX, 1, 1, 0, 3, 8)
# #    cv2.putText(result, left_curverad, (50,60), cv2.FONT_HERSHEY_SIMPLEX, 2, 255, 1)
#     text='%d' % 1 #left_curverad
#     cv2.putText(result, text, (640, 100), fontFace = cv2.FONT_HERSHEY_COMPLEX,
#                 fontScale = 2, color = (255, 0, 0), thickness = 2)
    return left_fit, right_fit