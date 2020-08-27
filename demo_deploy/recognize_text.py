import pytesseract
import numpy as np
import cv2
import os
from preprocessing_image import TransformImage,ImageFilteringConversion 
import text_processing as txp
from PIL import Image
import PIL
import core_text_log
import logging
logger = logging.getLogger('core_text.log')

# this module/class is an integration of small class methods in a required steps to accomplish ocr and cleaning text
class TextRecognition:
    def __init__(self):
        pass
    # classmethod passes images and language for Tesseract engine and returns raw extracted text with confidence of each extraced text. 
    #classmethod selects the os of machine to execute python without errors due to os dependency of Tesseract engine.
    # There are 4 oem configuration and 14 psm modes. --oem 1 and --psm 12 are best for our job with best results. It has LSTM embedded to extract text line by line.
    @classmethod
    def ocr_engine(cls,image,language='eng'):
        custom_config = r'--oem 1 --psm 12'#12
        if os.name=='nt':  # if os in machine is windows follow the if block statements with ocr engine
            logger.info('Successfully running on {}'.format(os.name))
            try:
                pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
                TESSDATA_PREFIX = 'C:/Program Files /Tesseract-OCR'
                strings=pytesseract.image_to_data(image,config=custom_config,output_type=pytesseract.Output.DICT,lang=language,timeout=6)# timeout=6 means only 6 secs time tessetact works
                text=strings.splitlines()
            except OSError as e:
                logger.error('Caught {}'.format(e))
                raise e
            except MemoryError as e:
                logger.critical('Caught {}'.format(e))
                raise e
            except RuntimeError as e:
                logger.error('Caught {}'.format(e))
                raise e
            except Exception as e:
                logger.error('Caught {}'.format(e))
                raise e
        elif os.name=='posix': # if os in machine is mac/linux then follows the if block statements with ocr.
            logger.info('Successfully running on {}'.format(os.name))
            try:
                strings=pytesseract.image_to_data(image,config=custom_config,output_type=pytesseract.Output.DICT,lang='eng')
                text=strings
            except OSError as e:
                logger.error('Caught {}'.format(e))
                raise e
            except MemoryError as e:
                logger.critical('Caught {}'.format(e))
                raise e
            except RuntimeError as e:
                logger.error('Caught {}'.format(e))
                raise e
            except Exception as e:
                logger.error('Caught {}'.format(e))
                raise e
        else:
            logger.error('OS not supported: {}'.format(os.name))
            raise OSError(' Not supported OS. Please choose linux/mac or windows')
        return text

    # This classmethod preprocesses the given image from validity check of image to all filtering and image transformation processes in specific procedure/sequence of classmethods.
    @classmethod
    def preprocess_image_for_ocr(cls,image_object):
        image_undo_perspective=TransformImage.perspective_transform(image_object,img_scale=False)
        im_align=TransformImage.text_align_horizontal(image_undo_perspective)
        im_normalize=ImageFilteringConversion.remove_shadow_normalize(im_align)
        im_to_gray=ImageFilteringConversion.color_gray(im_normalize)
        #im=ImageFilteringConversion.adaptive_constrast(im)#only for testing
        im_smoothing=ImageFilteringConversion.bilateral_smooth_image(im_to_gray)
        im_thresholding=ImageFilteringConversion.thresholding(im_smoothing)
        im_invert=ImageFilteringConversion.invert_bitwise_image(im_thresholding)
        im_morph_open=ImageFilteringConversion.morphological_filtering('open',im_invert,ker_size=1,iters_freq=2)# cleans outside dots noise of around char
        im_morph_close=ImageFilteringConversion.morphological_filtering('close',im_morph_open,ker_size=1,iters_freq=2)# fills black dots inside char
        im_invert_black_text=ImageFilteringConversion.invert_bitwise_image(im_morph_close)
        cleaned_image=ImageFilteringConversion.adaptive_constrast(im_invert_black_text)
        return cleaned_image

    #this classmethod first verifies  image, calls to preprocess image and finally passes cleaned and preprocessed image to ocr engine. Raw text is returned by this classmethod.
    @classmethod
    def get_strings_with_ocr(cls,image_input,card_type='Adhar',card_template={'f_name':'Rohit'},full_data=False):
        try:
            img_data=TextRecognition.preprocess_image_for_ocr(image_input)
            text=TextRecognition.ocr_engine(img_data)
            text_filter=txp.TextFilteration.filtering_raw_text(text,card_type,card_template,full_data)
            if type(text_filter)==dict:
                return text_filter
            else:
                logger.debug('output text {}'.format(text_filter))#print('log the errors')
        except Exception as e:
            logger.error('Caught {}'.format(e))
            raise e
        return None

    # this classmethod validates image and returns image if image path or image object is correct. Otherwise raises error or exceptions. If input image doesnt matches system standard, it returns error and dictionary of standard info.
    @classmethod
    def data_setup(cls,pil_image,file_extension,image_size):
        input_data_config={'min_image_size':'2KB','max_image_size':'200MB','image_resolution':(1200,1600),'file_extension':('.jpg','.jpeg'),'number_of_image':1,'Input_Error':''}
        input_error_face_image=dict()
        img=pil_image
        #check the size of image in bytes , if lies in between the range of standard size 
        img_size_lower_limit=300                       #size in bytes
        img_size_upper_limit=200000000                  #size in bytes
        min_image_shape=200                             # minimum image shape should be (200,200)
        max_image_shape=10000
        if  file_extension in ['.jpg','.jpeg']: 
            img_size=image_size #check image size in bytes
            if img_size<img_size_lower_limit or img_size>img_size_upper_limit:
                input_data_config['Input_Error']='The size of the Input image is < or > than, of {}'.format(img_size/(1024))
                input_error_face_image['ocr_input_error']=input_data_config
                logger.error('Error caused by size of image {}kB'.format(img_size/1024),stack_info=True,exc_info=True)
                return input_error_face_image

            #check whether the image input is RGB image format or not
            if len(img.getbands())<1:
                input_data_config['Input_Error']='Image is not 3 color channel: Invalid Image'
                input_error_face_image['ocr_input_error']=input_data_config
                logger.error('Error caused by Image is non RGB image input, {}'.format(img.getbands()),stack_info=True,exc_info=True)
                return input_error_face_image
            #check whether image has minimum resolution required  or not
            if max_image_shape>img.size[0]>min_image_shape and max_image_shape>img.size[1]>min_image_shape:
                #print('log it')
                pass
            else:
                input_data_config['Input_Error']='Not standard Image Resolution'
                input_error_face_image['ocr_input_error']=input_data_config
                logger.error('ValueError: Not standard Image Resolution',stack_info=True,exc_info=True)
                return input_error_face_image
        else:
            input_data_config['Input_Error']='Not standard Image File Extension'
            input_error_face_image['ocr_input_error']=input_data_config
            logger.error('ValueError: Not standard Image File Extension',stack_info=False,exc_info=False)
            return input_error_face_image
        return True

    @classmethod
    # orientation tag /rotation corrector
    def exif_transpose(cls,img):
        logger.debug('type of input for orientation  correction {}'.format(type(img)))
        if not img:
            return img
        exif_orientation_tag = 274
        # Check for EXIF data (only present on some files)
        if hasattr(img, "_getexif") and isinstance(img._getexif(), dict) and exif_orientation_tag in img._getexif():
            exif_data = img._getexif()
            orientation = exif_data[exif_orientation_tag]
            # Handle EXIF Orientation
            if orientation == 1:
                logger.debug('Normal image of orientation 1')
                # Normal image - nothing to do!
                pass
            elif orientation == 2:
                logger.debug('Mirrored left to right with orientation 2')
                # Mirrored left to right
                img = img.transpose(PIL.Image.FLIP_LEFT_RIGHT)
            elif orientation == 3:
                logger.debug('Incorrect 180 degre rotation with orientation 3')
                # Rotated 180 degrees
                img = img.rotate(180)
            elif orientation == 4:
                logger.debug('Upside down with orientation 4')
                # Mirrored top to bottom
                img = img.rotate(180).transpose(PIL.Image.FLIP_LEFT_RIGHT)
            elif orientation == 5:
                logger.debug('Mirrored top-left diagonal with orientation 5')
                # Mirrored along top-left diagonal
                img = img.rotate(-90, expand=True).transpose(PIL.Image.FLIP_LEFT_RIGHT)
            elif orientation == 6:
                logger.debug('Rotated -90 degrees  with orientation 6')
                # Rotated 90 degrees
                img = img.rotate(-90, expand=True)
            elif orientation == 7:
                logger.debug('Mirrored top-right diagonal with orientation 7')
                # Mirrored along top-right diagonal
                img = img.rotate(90, expand=True).transpose(PIL.Image.FLIP_LEFT_RIGHT)
            elif orientation == 8:
                logger.debug('Rotated 270 degrees with orientaion 8')
                # Rotated 270 degrees
                img = img.rotate(90, expand=True)
        return img

if __name__=='__main__':
    pass

