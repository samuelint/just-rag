import logging


logger = logging.getLogger(__name__)


class DecideToRetry:
    def __call__(self, state):
        retry_count = state.get("retry_count", 0)
        max_retry = state.get("max_retry", 1)

        if retry_count - 1 >= max_retry:
            logger.info("---MAX RETRY COUNT REACHED---")
            return "max_retry_count_reached"
        else:
            return "continue"
