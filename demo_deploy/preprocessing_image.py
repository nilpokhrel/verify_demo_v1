
import numpy as np
import cv2
import os
from skimage.filters import threshold_local
import core_text_log
import logging
logger = logging.getLogger('core_text.log')
# this class methods are meant to operate certain image filtering, conversions to image channels, and inversions
class ImageFilteringConversion:
    def __init__(self):
        pass
    #adaptive threshold to convert gray image to binary image, and returns processed binary image
    @classmethod
    def thresholding(cls,image):
        return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

    #joins the binary image and adds extra pixels to texts/black white pixels with defined kernel and iterations( times the dilation repeats), and returns processed image
    @classmethod
    def dilate(cls,image,kersize=1,iters_freq=1):
        kernel = np.ones((kersize,kersize),np.uint8)
        return cv2.dilate(image, kernel, iterations = iters_freq)#iter=1,kernwl=2,2

    #to a grayscale image it adjusts the constrast of brigntness and darkness of image, and returns processed image
    @classmethod
    def adaptive_constrast(cls,image):
        #print('Adapting constrast...')
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))#4.0,(16,16)
        image = clahe.apply(image)
        return image
    #converts rgb image  to gray image and returns gray image
    @classmethod
    def color_gray(cls,image):
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image=cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return image

    #converts white pixels to black and vice-versa
    @classmethod
    def invert_bitwise_image(cls,image):
        return cv2.bitwise_not(image)

    # smooths/ blurs image preserving the edges in the images, however it is slowest process in blur method
    @classmethod
    def bilateral_smooth_image(cls,image):
        return cv2.bilateralFilter(image,11,17,17)

    #blurs image with a kernel and keeps the median value of the convoluted image window
    @classmethod
    def median_blur(cls,image,kersize=3):
        return cv2.medianBlur(image,kersize, 0)

    #blurs image or smooth image with a kernel with given kernel size
    @classmethod
    def blur_image(cls,image,ker=1):
        return cv2.blur(cropped,(ker,ker))

    #finds edges in the objects in image with canny algorithm
    @classmethod
    def canny_image(cls,image,thres1=50,thres2=200):
        return cv2.Canny(image,thres1,thres2)
    # bluring image with gaussian function, its may create gaussian noises too
    @classmethod
    def gaussian_blur(cls,image,kersize=3):
        return cv2.GaussianBlur(image,(kersize,kersize), 0)

    # normalizes the shadowed image and  removes the shadow from the image in all color channels; improves recognition of text in images
    @classmethod
    def remove_shadow_normalize(cls,image):
        rgb_split=cv2.split(image)
        norm_image=[]
        for plane in rgb_split:
            dilate=cv2.dilate(plane,np.ones((3,3),np.uint8))#7,7
            blur_img=cv2.medianBlur(dilate,3)#21
            diff_img=255-cv2.absdiff(plane,blur_img)
            norm_img=cv2.normalize(diff_img,None,alpha=0,beta=255,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_8UC1)
            norm_image.append(norm_img)
        return cv2.merge(norm_image)

    #morphological operations can remove salt and pepper noises around the text and/or fills the black dots inside the bounded edge text with white pixels
    #open mode removes dots noises outside the text and close mode fills the black dots with white pixels
    #kernel size defines the size or window of the convolution
    #iter_freq is number of itertions to conduct convolution image with kernel sized window
    @classmethod
    def morphological_filtering(cls,mode,image,ker_size=1,iters_freq=1):
        kernel=np.ones((ker_size,ker_size),np.uint8)
        if mode=='open':
            filtered= cv2.morphologyEx(image,cv2.MORPH_OPEN,kernel,iterations=iters_freq)
        elif mode=='close':
            filtered=cv2.morphologyEx(image,cv2.MORPH_CLOSE,kernel,iterations=iters_freq)
        else:
            raise ValueError('Wrong Morphology mode.')
        return filtered

# this class of preprocessing image conducts some gerometrical transformations of image , aligning texts in image to horizontal with image transformations
class TransformImage:
    def __init__(self):
        pass

    #perspective transform method transforms perspective view to bird-eye-view
    @classmethod
    def perspective_transform(cls,image_object,img_scale=False):
        logger.debug('The type of image in perspective transform method from preprocessing_image is {}'.format(type(image_object)))
        img=image_object
        orig = img.copy()
        gray =ImageFilteringConversion.color_gray(img) #cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_blurred = ImageFilteringConversion.median_blur(gray,kersize=3)
        threshold_lower = 25#40
        threshold_upper = 200#150
        edged = ImageFilteringConversion.canny_image(gray_blurred,thres1=threshold_lower,thres2=threshold_upper)
        edged_copy = edged.copy()
        edged_copy = ImageFilteringConversion.gaussian_blur(edged_copy,kersize=3)
        (cnts, _) = cv2.findContours(edged_copy, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:4]
        screenCnt = []
        eplison=0.015
        scale=1 
        for c in cnts:
            # approximate the contour
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, eplison * peri, True)
            # approx = np.array(cv2.boundingRect(c))
            # if our approximated contour has four points, then we
            # can assume that we have found our target
            if len(approx) == 4:
                screenCnt = approx
                break
        if screenCnt.__len__()!= 0:
            warped_org_image = TransformImage.four_point_transform(orig, screenCnt.reshape(4, 2) * scale)
        else:
            warped_org_image = orig
        #below 3 lines of code are not used/implemented for this version due to not robust codes. Needs a lot of work for trial and test. However it improves better in transformation in unbounded image.

        #warped = cv2.cvtColor(warped_org_image, cv2.COLOR_BGR2GRAY)
        #warped = warped > threshold_local(warped, 251, offset=10)
        #warped = warped.astype('uint8') * 255
        return warped_org_image

    # aligns the text line in image horizontal after a sequence of following procedures
    #if alignments are performed tesseract correctly identifies the text in image, otherwise garbage chars are returned by tesseract
    #for better alignment and correct alignment the card must be clear and edges of rectangular card must be clear as possible. 
    #works fine if background of the image found black or dark and card with enough light
    # shadows must not be in image or spot light must not found in texts( since black texts becomes white if overlight /flash appears in image)
    @classmethod
    def text_align_horizontal(cls,image):
        #skew correction procedure
        orig_image=image.copy()
        image=ImageFilteringConversion.color_gray(image)# convert to gray and blur and threshold
        image=ImageFilteringConversion.median_blur(image,kersize=5)
        image=ImageFilteringConversion.thresholding(image)
        image=ImageFilteringConversion.invert_bitwise_image(image)
        image=ImageFilteringConversion.dilate(image,kersize=5,iters_freq=4)
        image=ImageFilteringConversion.invert_bitwise_image(image)

        coords = np.column_stack(np.where(image > 0))
        angle = cv2.minAreaRect(coords)[-1]
        scale=1
        #angle is calculated by rectangular box generated by the dilated texts in the image with white pixels(before calculation of white pixels the binary inversion is performed to convert black pixels in text to white)
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        logger.debug('The angle of text in image is {}'.format(angle))
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        # dont compute rotation if angle is too small since rotating image consumes time
        if (0.1> angle>-0.1):# angle+90 gives the actual slope of degree of text like ,0.0,2,-1
            return orig_image
        else:
            M = cv2.getRotationMatrix2D(center, angle,scale)
            rotated = cv2.warpAffine(orig_image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated

    #ordering the points of object detected with canny edge detection; to form an order of rectangular shape
    @classmethod
    def order_points(cls,pts):
        # order will be: top-left, top-right, bottom-right, bottom-left
        rect = np.zeros((4, 2), dtype='float32')

        # the top-left point will have the smallest sum,
        # the bottom-right point will have the largest sum
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        # the top-right point will have the smallest difference,
        # the bottom-left will have the largest difference
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    #takes image as input and four points of rectangular object detected in image to perform warpPerpective tranform 
    # calling this method with determined 4 points and image to be transformed returns original transformed (birds-eye-view image)
    @classmethod
    def four_point_transform(cls,image, pts):
        rect = TransformImage.order_points(pts)
        (tl, tr, br, bl) = rect

        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype='float32')
        #the height or width of detected retangular shape must be greater than 300 pixels (either height or width must not be less than 300 pixels)
        if  not(maxWidth<300 or maxHeight<300):
            M = cv2.getPerspectiveTransform(rect, dst)
            warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
            return warped
        return image

if __name__=='__main__':
    pass



































