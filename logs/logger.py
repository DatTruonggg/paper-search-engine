import loguru
from typing import Optional, TypeVar
from pathlib import Path


log_level = "DEBUG"
log_format = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS zz}</green> | "
    "<level>{level: <8}</level> | "
    "<yellow>Line {line: >4} ({file}):</yellow> <b>{message}</b>"
)

_T_logoru_logger = TypeVar("_T_logoru_logger", bound=loguru._logger.Logger)

def logger_handler(
    use_log_file: bool = True,
    file: Optional[str] = None,
) -> _T_logoru_logger:
    loguru.logger.remove() 

    loguru.logger.add(
        sink=lambda msg: print(msg, end=""),
        level=log_level,
        format=log_format,
        colorize=True,
        backtrace=True,
        diagnose=True,
    )

    if use_log_file:
        if file is None:
            file = f"./logs/logging_file.log"
        Path(file).parent.mkdir(parents=True, exist_ok=True)
        loguru.logger.add(
            file,
            level=log_level,
            format=log_format,
            colorize=False,
            backtrace=True,
            diagnose=True,
            rotation="1 week",
            retention="1 month",
        )
    return loguru.logger