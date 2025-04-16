from loguru import logger

def logging(func):
    def wrapper(*args, **kwargs):
        logger.info(f"Calling function {func.__name__}")
        result = func(*args, **kwargs)
        logger.info(f"Function {func.__name__} completed")
        return result
    return wrapper