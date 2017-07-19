import logging
import logging.handlers

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

rh=logging.handlers.TimedRotatingFileHandler('try.log','D')
fm=logging.Formatter("%(asctime)s  %(levelname)s - %(message)s","%Y-%m-%d %H:%M:%S")
rh.setFormatter(fm)
logger.addHandler(rh)

debug=logger.debug
info=logger.info
warn=logger.warn
error=logger.error
critical=logger.critical

a = 'hello'

info("testlog1"+a)
warn("warn you %s","costaxu")
critical("it is critical")