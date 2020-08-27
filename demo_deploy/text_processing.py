import re
import core_text_log
import logging
logger = logging.getLogger('core_text.log')
class TextFilteration:
    def __init__(self):
        pass
    # this method filters raw and noisy texts to a clean comparable text for matching text from card to the template values,
    #  and returns a dictionary of matched text with confidence and similarity ratio
    @classmethod
    def filtering_raw_text(cls,extracted_text,card_type='Adhar',card_template={'f_name':'Rohit'},full_data=False):
        logger.debug('Input to filtering_raw_text method from text_processing module: text {}, card_type {}, card_template {}, full_data {}'.format(type(extracted_text),card_type,card_template,full_data))
        if len(extracted_text)!=0:
            text=extracted_text
        else:
            logger.error('Found empty strings in text extracted from Tesseract: {}'.format(len(extracted_text)))
            raise ValueError('Found empty string extracted from Tesseract')
        # confidence of text detection extracted from ocr results and text
        conf_list=text['conf']
        text_list=text['text']
        card_info=[]
        output=dict()
        #if type(card_type)==str and type(card_template)==dict:
        # if full_data==True returns with full template, else returns datas relating to keys of template
        for i in range(len(conf_list)):
            # first level filtering text with confidence of the words/texts extracted with tesseract and removing special characters using regex( alphabets and numbers strings that had more than 1 character)
            # text with only one char is meaningless for our requirement so filtered at very first of filtering process.
            if int(conf_list[i])>45 and len(text_list[i])>1:
                text=re.sub(r'[^a-zA-Z0-9]',' ',text_list[i])
                card_info.append(text)
        output['text_result']=card_info
        return output
if __name__=='__main__':
    pass
