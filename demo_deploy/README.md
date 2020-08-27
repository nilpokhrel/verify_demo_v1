The text recognition system has dependency on platform( os dependency) for Tesseract. It operates with some extra configuration which is defined in recognize_text module. It automatically executes ocr engine based on os in machine using python built-in os module. Rest of the modules are platform independent, however required modules are to be installed successfully to run the system.

The requirements with versions are well written in requirements.txt file. Tesseract is a c++ library so python wrapper is necessary to successfully run the system by module as named pytesseract.

"""
To Call class method for OCR successively:
    text=TextRecognition.get_strings_with_ocr(input_image)
    This get_strings_with_ocr calls image validity check and later calls the image processing processes in s successive manner.
Then, the raw text with garbages must be cleaned.
    text_filter=txp.TextFilteration.filtering_raw_text(text,card_type='pan',card_template=template,full_data=False) 

    filtering_raw_test() class method has 4 arguments to be passed as per requirements. Text is the raw text extracted from OCR engine. Card_type has None as default. But if given card type of and is registered in system with template then returns the dictionary with requested key and values of the template, matched ratio and confidence. The last full_data argument is False in default. If True then the method returns the dictionary with all the text in the card by calling utility function to generate full template like below:

    suppose input template is =>  template={'f_name':'Rohit','l_name':'Kumar','dob':'10/01/1982'}
    then output will have some extra key values which are pre-defined in pan_card and adhar_card modules.
    output_dict={'title1':'Government','title2':'of','title3':'India','f_name':'Rohit','m_name':'','l_name':'Kumar','dob':'10/01/1982'} 
    Date format can with other special charactes too, like 10-01-1982.
    Note: In practice full_data is set to False.
"""

For function API or using this text recognition to Web services or API services recognize_text module(two function calls) can be called with some required input to the system. And the output of system can be used for response by API. Below is the input and output examples:
1. Input: 
    image object or path to image 
    Template or information that is to be compared with the text in the image/card.
    The type of card should be passed to the system, otherwise cleaning may not be obtained as desired. Card and template can be passed like this (text,card_type='pan',card_template=template,full_data=False).
    Template has a predefined keys and must be included in query dictionary like this:
    template={'f_name':'ROHIT','dob':'10.01.1982','l_name':'KUMAR','fa_f_name':'RAJIV','pin':'DMLPK6157E'} #for pan

2. Output of the system is a dictionary with key value pairs. The values is a list of text , confidence of text detection and string match ratio between text with value of same key.
    Output format will be like this with key: text_result
    {text_result:{'f_name': ['ROHIT', 85, 1.0], 'l_name': ['KUMAR', 95, 1.0], 'fa_f_name': ['RAJIV', 79, 1.0], 'dob': ['10071982', 87, 0.875], 'pin': ['DIMLPK6157E', 38, 0.952]}}
    key of a dictionary is what we need to get from image text, and values is a list of identified and detected text from image.
    If error caused due to input image with different in standard of image, then a error dictionary is returned with necessary content and key: data_config_error

Main classmethod to be called:

1. text=recognize_text.TextRecognition.get_strings_with_ocr(input_image)
2. text_filter=text_processing.TextFilteration.filtering_raw_text(text,card_type='pan',card_template=template,full_data=False) 
