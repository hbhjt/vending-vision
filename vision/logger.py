"""
日志模块

基于 Python logging 的日志配置，支持：
- 文件日志（按大小轮转）
- 控制台输出
- 统一的日志格式
"""

import logging
import os
from logging.handlers import RotatingFileHandler

from vision.config import settings


def setup_logger():
    """配置并返回 vending_vision 应用的全局日志记录器。

    特性：
    - 同时输出到文件和控制台
    - 文件日志按大小自动轮转（默认 5MB，保留 5 个备份）
    - 日志级别可通过配置控制（LOG_LEVEL）
    - 日志目录自动创建
    """
    os.makedirs(settings.LOG_DIR, exist_ok=True)

    logger = logging.getLogger("vending_vision")
    level = getattr(logging, str(settings.LOG_LEVEL).upper(), logging.INFO)
    logger.setLevel(level)

    # 防止重复添加处理器
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 文件日志处理器（按大小轮转）
    file_handler = RotatingFileHandler(
        settings.LOG_FILE,
        maxBytes=max(int(settings.LOG_MAX_BYTES), 0),
        backupCount=max(int(settings.LOG_BACKUP_COUNT), 0),
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    # 控制台日志处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# 全局日志记录器实例
logger = setup_logger()
