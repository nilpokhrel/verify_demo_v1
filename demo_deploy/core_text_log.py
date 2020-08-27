import logging
logging.basicConfig(filename='core_text.log',filemode='w',format='%(asctime)s %(module)s:%(lineno)s %(funcName)s -%(name)s - %(levelname)s - %(message)s',level=logging.DEBUG)#logging.FATAL
'''
# Create a custom logger
logger = logging.getLogger(__name__)

# Create handlers
c_handler = logging.StreamHandler()
f_handler = logging.FileHandler('core.log',mode='w')
c_handler.setLevel(logging.WARNING)
c_handler.setLevel(logging.DEBUG)
c_handler.setLevel(logging.CRITICAL)
f_handler.setLevel(logging.ERROR)

# Create formatters and add it to handlers
c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
c_handler.setFormatter(c_format)
f_handler.setFormatter(f_format)

# Add handlers to the logger
logger.addHandler(c_handler)
logger.addHandler(f_handler)

logger.warning('This is a warning')
logger.error('This is an error in code')
'''
