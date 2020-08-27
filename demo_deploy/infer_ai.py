import torch
from PIL import Image 
import PIL
from scipy.spatial import distance
import numpy as np
from facenet_pytorch import MTCNN
import os
import logging
import core_face_verify
logger = logging.getLogger('core_face_verification.log')

# a defined class FaceVerification where all necessary functions/methods are bunndled for clarity and reusable. Every functions are tested
# individually with most of the favourable and unfavourable test cases.
class FaceVerification:
    def __init__(self):
        pass
    # datasetup takes input as two directory paths,validates directory,validates image file type, counts the number of images in source path, checks minimum image shape
    # If everything is ok it calls mtcnn detection for face alignment and crops face from image
    # used to compute coordinate of eyes and compute angle (inclination of face with respect of image coordinates). After align process function extracts the face image
    # and creates a subdirectory in destination directory with this filename. These subdirectories are stored with respective filename images. 
    # while saving image image to destination subdirs it checks whether the cropped face has minimum face resolution or not. If not returns error dictionary instead of integer count of images
    @classmethod
    def check_face_size(cls,images):
        input_data_config={'min_image_size':'2KB','max_image_size':'200MB','image_resolution':(1200,1600),'file_extension':('.jpg','.jpeg'),'min_number_image':2,'minimum_face_resolution':(60,60),'Input_Error':''}
        if images.size[0]<60 or images.size[1]<60:
            input_data_config['Input_Error']='The resolution of face in image is too small of size {}'.format(images.size)
            input_error_face_image['face_input_error']=input_data_config
            logger.error('ValueError: The resolution of face in image is too small of size {}'.format(images.size),stack_info=False,exc_info=True)
            return input_error_face_image
        else:
            return True
    @classmethod
    def data_setup(cls,pil_image,file_extension,image_size):
            input_data_config={'min_image_size':'2KB','max_image_size':'200MB','image_resolution':(1200,1600),'file_extension':('.jpg','.jpeg'),'min_number_image':2,'minimum_face_resolution':(60,60),'Input_Error':''}
            input_error_face_image=dict()
            img=pil_image
            #check the size of image in bytes , if lies in between the range of standard size 
            img_size_lower_limit=2000                       #size in bytes
            img_size_upper_limit=200000000                  #size in bytes
            min_image_shape=200                             # minimum image shape should be (200,200)
            max_image_shape=10000

            if  file_extension in ['.jpg','.jpeg']: 
                img_size=image_size #check image size in bytes
                if img_size<img_size_lower_limit or img_size>img_size_upper_limit:
                    input_data_config['Input_Error']='The size of the Input image is < or > than, of {}'.format(img_size/(1024))
                    input_error_face_image['face_input_error']=input_data_config
                    logger.error('Error caused by size of image {}kB'.format(img_size/1000),stack_info=True,exc_info=True)
                    return input_error_face_image

                #check whether the image input is RGB image format or not
                if len(img.getbands())!=3:
                    input_data_config['Input_Error']='Image with monocolor/not RGB is Invalid Image'
                    input_error_face_image['face_input_error']=input_data_config
                    logger.error('Error caused by monocolor/not RGB channel of image, {}'.format(img.getbands()),stack_info=True,exc_info=True)
                    return input_error_face_image
                #check whether image has minimum resolution required  or not
                if img.size[0]>min_image_shape and img.size[0]<max_image_shape and img.size[1]>min_image_shape and img.size[1]<max_image_shape:
                    pass
                else:
                    input_data_config['Input_Error']='Not standard Image Resolution'
                    input_error_face_image['face_input_error']=input_data_config
                    logger.error('ValueError: Not standard Image Resolution',stack_info=True,exc_info=True)
                    return input_error_face_image
            else:
                input_data_config['Input_Error']='Not standard Image File Extension'
                input_error_face_image['face_input_error']=input_data_config
                logger.error('ValueError: Not standard Image File Extension',stack_info=False,exc_info=False)
                return input_error_face_image
            return True



    # align with mtcnn, detect faces in images and  crop face from image after rotating image about center of eyes.
    @classmethod
    def rotate_image(cls,image,mtcnn): 
        logger.debug('Input to rotate_image type: {}'.format(type(image)))
        mtcnn = mtcnn
        status=mtcnn.detect(image,landmarks=True)
        _,prob,facemarks=status
        #check all elements of probability list that are not none
        if prob.all() is None:
            logger.error('Found None in probability of face detection process: Face not found in image.')
            raise ValueError('Found None in probability of face detection process: Face not found in image')
        if facemarks.all() is not None:#all facemarks[0.1,0.3...] values must be true
            if  prob.any()>0.989: #.any finds at least one values from array 
                el=facemarks[0][0] # point of left eye in image
                er=facemarks[0][1] #point of right eye in image
                nose=facemarks[0][2] # point of nose tip in image
                ml=facemarks[0][3] #point of left point of mouth in image
                mr=facemarks[0][4] # point of right point of mouth in image
                dy=er[1]-el[1] #difference in vertical distance in y axis
                dx=er[0]-el[0] # difference in horizontal distance in x-axis
                dist_eyes=np.sqrt(dy**2+dx**2)
                angle=np.degrees(np.arctan2(dy,dx)) # angle is determined by the slope of a line between two detected eyes center point with reference to horizontal x-axis
                xmid=(er[0]+el[0])/2 #for x
                ymid=(er[1]+el[1])/2
                eyescenter=(xmid,ymid) # center coordinate in between two eyes
                # skilp angle in between as below to reduce computation cost od device
                if angle>0.09 or angle<-0.09:
                    rotated_image=image.rotate(angle,expand=0,center=eyescenter,translate=None)
                else:
                    rotated_image=image

                #cropping the face from image
                dist_eyes=int(dist_eyes)                                    # dist_eyes is the distance between two eyes center point
                xmid=int(xmid)
                ymid=int(ymid)
                img=np.asarray(rotated_image)
                h=img.shape[0]; w=img.shape[1]                              # h is the height of image given and w is the width of the given image
                x1=xmid-int(1.2*dist_eyes); y1=ymid-int(1.3*dist_eyes)      # x1 and x2 is on horizontal x-axis and y1 and y2 is on vertical y-axis of the given image to crop face
                x2=xmid+int(1.2*dist_eyes); y2=ymid+int(2.2*dist_eyes)

                #search the face from image and crop within the image border with proper check
                if x1>0 and y1>0:                                           # if left eye in face lies near the left border of image and center of eyes lies atleast 1.18 times the distance betn eyes up forehead
                    if x2<w+1 and y2<h+1:
                        img=img[y1:y2,x1:x2,:]
                        im=Image.fromarray(img)
                        return im
                    elif x2>w and y2<h:
                        x2=w
                        img=img[y1:y2,x1:x2,:]
                        im=Image.fromarray(img)
                        return im
                    elif x2<w+1 and y2>h:
                        y2=h
                        img=img[y1:y2,x1:x2,:]
                        im=Image.fromarray(img)
                        return im
                    elif x2>w and y2>h:
                        x2=w
                        y2=h
                        img=img[y1:y2,x1:x2,:]
                        im=Image.fromarray(img)
                        return im
                    else:
                        logger.error('Couldnt align and crop face for {}'.format(filename))
                        raise ValueError(' Error while aligning and cropping face in mtcnn')

                elif x1<1 or y1<1 or x2>w:                                  # if left eye or right eye or center of eye are closer to border of image
                    if x2>w:
                        x2=w
                    if y2>h:
                        y2=h
                    if y1<1:
                        y1=0
                    if x1<1:
                        x1=0
                    img=img[y1:y2,x1:x2,:]
                    im=Image.fromarray(img)
                    return im
                else:
                    return rotated_image
            else:
                logger.error('Image has no human face in {}'.format(filename))
                raise ValueError('Image contains no real face.')
        else:
            logger.error('Image has no face {}'.format(filename))
            raise ValueError('Images detected with None faces.')
        return False

    #images to mtcnn model which detects images and later the image tensor is stacked in list for embedding computation
    '''
    error :RuntimeError: There were no tensor arguments to this function (e.g., you passed an empty list of Tensors), but no fallback function is registered for schema aten::_cat.  
    This usually means that this function requires a non-empty list of Tensors.  Available functions are [CUDATensorId, CPUTensorId, VariableTensorId
    Caused by when image resolution is too low than setting 'min_face_size= integer_value' in MTCNN initialization in hardware setup
    '''
    @classmethod
    def stack_tensor_mtcnn_model(cls,pil_image,mtcnn):                    # mtccn application for test images to detect face in image and stack image tensors
        logger.debug('Input (destination directory) path to stack tensor mtcnn resnet model: '.format(type(pil_image)))
        mtcnn = mtcnn
        try:
            detected_x, prob = mtcnn(pil_image, return_prob=True)
            logger.debug(' Mtcnn detected face in image: {} and probability {}'.format(type(detected_x),prob))
        except MemoryError as e:
            logger.critical(e,stack_info=True,exc_info=True)
            raise e
        except RuntimeError as e:
            logger.error(e,stack_info=True,exc_info=True)
            raise e
        except RecursionError as e:
            logger.error(e,stack_info=True,exc_info=True)
            raise e
        except Exception as e:
            logger.error(e,stack_info=True,exc_info=True)
            raise e
        if detected_x is not None:
            if prob.any()>0.989:
                img_detected=detected_x
            else:
                logger.error('Probability {} of face detection is below required in image.'.format(prob))
                raise ValueError('Probability of face detection is below required.')
        else:
            logger.error('Face not found in image')
            raise ValueError('Face not found.')
        return img_detected

    @classmethod
    def resnet_model(cls,resnet_model_object,device,image_stack):
        resnet=resnet_model_object
        try:
            aligned_stack=torch.stack(image_stack).to(device)
            embeddings=resnet(aligned_stack).detach().cpu()
        except MemoryError as e:
            logger.critical('{} caught in resnet and torch.stack'.format(e),stack_info=True,exc_info=True)
            raise e
        except RuntimeError as e: #appears also when theres no human face in image
            logger.error('{} caught in resnet and torch.stack'.format(e),stack_info=True,exc_info=True)
            raise e
        except Exception as e:
            logger.error('{} caught in resnet and torch.stack'.format(e),stack_info=True,exc_info=True)
            raise e
        return embeddings

    # cosine distance function calculates distances among or between the embeddings computed and returned by stack tensor mtcnn function
    #And it returns the computed pandas dataframe of distances and names of images , or returns distances only
    @classmethod
    def cosine_distance(cls,embeddings):
        logger.debug('Input type embeddings and names to cosine_distance: {} '.format(type(embeddings)))
        dists=distance.cosine(embeddings[0],embeddings[1])
        return dists

    #function takes pandas dataframe and names of images ,threshold to verify images on  mode basis of verification(pair match,multimatch) and return (tuple for pairmatch and dataframe for multimatch)
    #The actual output of the face verification system is returned by this method as a dictionary for pair_match and pandas dataframe for multi_match
    @classmethod
    def verify_faces(cls,dist,thres):
        logger.debug('Input to verify_faces method: thres {}'.format(thres))
        thres_hold=thres # for verification to apply as classifier 
        output_text_result=[]
        if dist<thres_hold and dist>0.0:
            output_text_result.append((True,thres_hold))
            return output_text_result
        elif dist>thres_hold or dist==thres_hold:
            output_text_result.append((False,thres_hold))
            return output_text_result 
        else:
            output_text_result.append((True,thres_hold))
            return output_text_result
        return None
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
