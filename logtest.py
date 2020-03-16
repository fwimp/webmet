import logging
from webmet import *
from webmet.exceptions import logtest_exceptions

logger = logging.getLogger('webmet')
logger.setLevel(logging.DEBUG)

# Set custom logging handler to test whether overriding the NullHandler is working
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(funcName)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

logtest_merge()
logtest_digitise()
logtest_exceptions()
