import logging

logger = logging.getLogger(__name__)

def autopatch_all():
    try:
        import rlboost.sglang.autopatch
    except Exception as e:
        logger.debug(f"Failed to import rlboost.sglang.autopatch: {e}")

autopatch_all()
