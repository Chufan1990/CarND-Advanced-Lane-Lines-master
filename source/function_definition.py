import numpy as np
import cv2

#  fuction definition


def abs_sobel_thresh(img, orient='x', thresh=(0, 255)):
    # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = img    
    if orient == 'x':
        sobel = np.fabs(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    elif orient == 'y':
        sobel = np.fabs(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    abssobel = np.uint8(255 * sobel / np.max(sobel))
    mask = np.zeros_like(abssobel)
    mask[(abssobel >= thresh[0]) & (abssobel <= thresh[1])] = 1
    return mask


def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Convert to grayscale
    # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = img
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag) / 255
    gradmag = (gradmag / scale_factor).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return binary_output


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi / 2)):
    # Grayscale
    # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = img
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output = np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output


def hls_select(img, thresh=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:, :, 2]
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
    return binary_output


def img_roi(img):
    xsize = img.shape[1]
    ysize = img.shape[0]
    region_select = np.copy(img)
    left_bottom = [250, ysize - 100]
    right_bottom = [xsize - 250, ysize - 100]
    apex1 = [xsize / 2 - 40, ysize / 2 + 70]
    apex2 = [xsize / 2 + 40, ysize / 2 + 70]
    fit_left = np.polyfit(
        (left_bottom[0], apex1[0]), (left_bottom[1], apex1[1]), 1)
    fit_bottom = np.polyfit(
        (left_bottom[0], right_bottom[0]), (left_bottom[1], right_bottom[1]), 1)
    fit_right = np.polyfit(
        (right_bottom[0], apex2[0]), (right_bottom[1], apex2[1]), 1)
    fit_up = np.polyfit((apex1[0], apex2[0]), (apex1[1], apex2[1]), 1)
    XX, YY = np.meshgrid(np.arange(0, xsize), np.arange(0, ysize))
    region_thresholds = (YY < (XX * fit_left[0] + fit_left[1])) | \
        (YY < (XX * fit_right[0] + fit_right[1])) | \
        (YY > (XX * fit_bottom[0] + fit_bottom[1])) | \
        (YY < (XX * fit_up[0] + fit_up[1]))
    region_select[region_thresholds] = 0
    return region_select


def img_warp(img):
    img_size = (img.shape[1], img.shape[0])
    src = np.float32([[240, 719], [579, 450], [712, 450], [1165, 719]])
    dst = np.float32([[300, 719], [300, 0], [900, 0], [900, 719]])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    Minv = cv2.getPerspectiveTransform(dst, src)
    return warped, M, Minv


def lane_find(img):
    xsize = img.shape[1]
    ysize = img.shape[0]
    region_select = np.copy(img)
    histogram = np.sum(img[360:, :], axis=0)
    out_img = np.dstack((img, img, img)) * 255
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    nwindows = 9
    window_height = np.int(img.shape[0] / nwindows)
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    leftx_current = leftx_base
    rightx_current = rightx_base
    margin = 100
    minpix = 50
    left_lane_inds = []
    right_lane_inds = []
    for window in range(nwindows):
        win_y_low = img.shape[0] - (window + 1) * window_height
        win_y_high = img.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (
            nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (
            nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
    # y_eval = np.max(ploty)
    # ym_per_pix = 30 / 720
    # xm_per_pix = 3.75 / 700
    # left_fit_cr = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
    # right_fit_cr = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)
    # left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix +
    #                        left_fit_cr[1])**2)**1.5) / np.absolute(2 * left_fit_cr[0])
    # right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix +
    #                         right_fit_cr[1])**2)**1.5) / np.absolute(2 * right_fit_cr[0])
    return left_fit, right_fit #, left_curverad, right_curverad


def coor_convert(left_fit, right_fit, xsize, ysize, y_input=(100, 200), xscale=1, yscale=1):
    x_left = [(left_fit[0] * y_input[0]**2 + left_fit[1] * y_input[0] + left_fit[2]),
              (left_fit[0] * y_input[1]**2 + left_fit[1] * y_input[1] + left_fit[2])] - xsize / 2 * xscale
    x_right = ([(right_fit[0] * y_input[0]**2 + right_fit[1] * y_input[0] + right_fit[2]),
                (right_fit[0] * y_input[1]**2 + right_fit[1] * y_input[1] + right_fit[2])] - ysize) * -1 * yscale
    return x_left, x_right
    
