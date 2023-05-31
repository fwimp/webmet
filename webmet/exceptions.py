import logging
logger = logging.getLogger(__name__)
logger.propagate = True
def logtest_exceptions():
    s = "Exceptions logger"
    logger.critical(s)
    logger.error(s)
    logger.warning(s)
    logger.info(s)
    logger.debug(s)

class WebKernelError:
    pass

class MergeError(WebKernelError):
    pass