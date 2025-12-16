import os
import importlib
import importlib.util

if os.getenv("ENABLE_RLBOOST_AUTOPATCH", "false").lower() in ("true", "1"):
    if importlib.util.find_spec("rlboost.autopatch") is not None:
        importlib.import_module("rlboost.autopatch")


