#face verification imports
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from flask import Flask, flash, request, jsonify,make_response
from PIL import Image
import logging, os
from infer_ai import FaceVerification as fv
import numpy as np
import os
from flask_restful import Api,Resource
from flask_httpauth import HTTPBasicAuth

#ocr import modules
import text_dir.recognize_text as recog


#initializing flask app and api
app = Flask(__name__)
api=Api(app)
auth = HTTPBasicAuth()
file_handler = logging.FileHandler('server.log',mode='w')
app.logger.addHandler(file_handler)
app.logger.setLevel(logging.DEBUG)
USER_DATA={"admin":"@intelligent_universe"}
@auth.verify_password
def verify(username,password):
    app.logger.debug('username {} and password {}'.format(username,password))
    if not (username and password):
        return False
    return USER_DATA.get(username)==password

@auth.error_handler
def unauthorized():
    # return 403 instead of 401 to prevent browsers from displaying the default
    app.logger.debug('unathorized')
    return make_response(jsonify({'message': 'Unauthorized access'}), 403)

#loading faceverification deep network models
try:
    #select type of device available in machine
    device = torch.device(cuda if torch.cuda.is_available() else 'cpu')
    #evaluate resnet model
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    mtcnn=MTCNN(image_size=160, margin=0, min_face_size=60,select_largest=True,thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,device=device)
    app.logger.debug('Models loaded successfully: device,resnet,mtcnn')
except Exception as e:
    app.logger.debug('Exception caugnt at Models {}'.format(e))

#threshold for face verification
THRES=0.38

#post api method class for face verification
class face_match(Resource):
    @auth.login_required
    def post(self):
        # check if the post request has the file part
        app.logger.debug(' -------------------------FACE VERIFICATION LOG-------------------------------------------')
        app.logger.debug(request)
        if len(request.files)!=2:
            app.logger.debug('number of request files {}'.format(len(request.files)))
            return jsonify({'Number of images to verify face match is: ':2})
        try:
            file1 = request.files['file1']
            file2 = request.files['file2']
        except Exception as e:
           app.logger.error('Exception caught {}'.format(e))
           return jsonify({'Error processing request image files': 500})

        image1_extension= os.path.splitext(file1.filename)[1].lower()#file1.rsplit('.', 1)[1].lower()
        image2_extension= os.path.splitext(file2.filename)[1].lower()#file2.rsplit('.', 1)[1].lower()
        app.logger.debug('file1 extension and file2 extension {} {}'.format(image1_extension,image2_extension))
        if image1_extension and image2_extension in ['.jpg','.jpeg']:
            image_size1=len(file1.read())
            image_size2=len(file2.read())
            app.logger.debug('Size of files: {} KB and {} KB'.format(image_size1/1024,image_size2/1024))
            try:
                image1 = Image.open(file1)
                image2 = Image.open(file2)
            except Exception as e:
                app.logger.error('Caught error reading files: {}'.format(e))
                return jsonify({'cannot read file: ':'Unable to read file'})#,str(e)})

            verify_image1=fv.data_setup(image1,image1_extension,image_size1)
            verify_image2=fv.data_setup(image2,image2_extension,image_size2)
            app.logger.debug('Image verifications: {} {}'.format(verify_image1,verify_image2))
        else:
            app.logger.debug('file extension error {} {}'.format(image1_extension,image2_extension))
            return jsonify({'file_extension_error: ':(image2_extension,image1_extension)})

        if verify_image1==True and verify_image2==True:
            try:
                img1=fv.exif_transpose(image1)
                img2=fv.exif_transpose(image2)
                img1_rotated=fv.rotate_image(img1,mtcnn)
                img2_rotated=fv.rotate_image(img2,mtcnn)
            except Exception as e:
                app.logger.error('Caught exception {}'.format(e))

            bool_img1=fv.check_face_size(img1)
            bool_img2=fv.check_face_size(img2)
            app.logger.debug('Face sizes in image: {} and {}'.format(bool_img1,bool_img2))
            if bool_img1==True and bool_img2==True:
                stack_image=[]
                try:
                    img1_tensor=fv.stack_tensor_mtcnn_model(img2,mtcnn)
                    img2_tensor=fv.stack_tensor_mtcnn_model(img1,mtcnn)
                except Exception as e:
                    app.logger.error('caught error: {}'.format(e))
                stack_image.append(img1_tensor)
                stack_image.append(img2_tensor)
                try:
                    embeddings=fv.resnet_model(resnet,device,stack_image)
                except Exception as e:
                    app.logger.error('caught error: {}'.format(e))

                dist=fv.cosine_distance(embeddings)
                app.logger.debug('The similarity distance is : {}'.format(dist))
                result=fv.verify_faces(dist,THRES)
                verify,Threshold=result[0]
                return jsonify({'face_result':{'similarity_distance':dist,'Threshold':Threshold,'face_match':verify}})
            else:
                if bool_img1==True:
                    err=bool_img2
                err=bool_img1
                return jsonify({'file_standard_error':err})
        else:
            if verify_image1==True:
                err=verify_image2
            err=verify_image1
            return jsonify({'file_standard_error':err})


# text ocr api
class Text_verification(Resource):
    @auth.login_required
    def post(self):
        app.logger.debug('----------------------------------------TEXT RECOGNITION LOG-----------------------------------------------')
        app.logger.debug(request)
        if len(request.files)!=1:
            app.logger.debug('Number of files ocr: {}'.format(len(request.files)))
            return jsonify({'Number of images to for OCR is: ':1})
        try:
            file1 = request.files['file1']
        except Exception as e:
           app.logger.error('Error handling request file in ocr {}'.format(e))
           return jsonify({'Error processing request image files': 500})

        image1_extension= os.path.splitext(file1.filename)[1].lower()
        app.logger.debug('File extension in ocr: {}'.format(image1_extension))
        if image1_extension in ['.jpg','.jpeg']:
            image_size1=len(file1.read())
            app.logger.debug('File size in ocr: {} KB'.format(image_size1/1024))
            try:
                image1 = Image.open(file1)
            except Exception as e:
                app.logger.error('caught error in ocr opening image: {}'.format(e))
                return jsonify({'cannot read file: ':'Unable to read file'})#,str(e)})

            verify_image1=recog.TextRecognition.data_setup(image1,image1_extension,image_size1)
            app.logger.debug('Image verification in ocr {}'.format(verify_image1))
        else:
            return jsonify({'file_extension_error: ':(image1_extension,)})

        if verify_image1==True:
            try:
                img1=recog.TextRecognition.exif_transpose(image1)
                img_ocr=np.array(img1)
                text=recog.TextRecognition.get_strings_with_ocr(img_ocr)
                if text['text_result']:
                    return jsonify({'orc_result':text['text_result']})
                elif text['ocr_input_error']:
                    return jsonify({'ocr_input_error':text['ocr_input_error']})
                else:
                    app.logger.debug(' Output of ocr: {}'.format(text))

            except Exception as e:
                app.logger.error('Error caught {}'.format(e))
        else:
            return jsonify({'ocr_input_error':verify_image1['ocr_input_error']})

api.add_resource(face_match,'/demo-face_match')
api.add_resource(Text_verification,'/demo-ocr')
if __name__ == '__main__':
    app.run(host='0.0.0.0',port=80,debug=False)
























