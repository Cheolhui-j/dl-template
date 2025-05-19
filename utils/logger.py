import logging

def get_logger(log_file):
    logger = logging.getLogger("trainer")
    logger.setLevel(logging.INFO)

    # 파일 핸들러
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)

    # 콘솔 핸들러
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # 포맷 설정
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger