import logging
import sys


def setup_logging(debug: bool = False) -> logging.Logger:
    """Configures structured application logging."""
    log_level = logging.DEBUG if debug else logging.INFO
    log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Silence noisy loggers
    logging.getLogger("uvicorn.access").setLevel(logging.INFO)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    
    logger = logging.getLogger("lungvision")
    logger.setLevel(log_level)
    return logger


logger = setup_logging()

