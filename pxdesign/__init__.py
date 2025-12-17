import os

if os.environ.get("PROTENIX_DATA_ROOT_DIR") is None:
    default_value = os.path.abspath("./release_data/ccd_cache")
    os.environ["PROTENIX_DATA_ROOT_DIR"] = default_value
    # print(f"Environment variable PROTENIX_DATA_ROOT_DIR not set. Using default: {default_value}")
if os.environ.get("TOOL_WEIGHTS_ROOT") is None:
    default_value = os.path.abspath("./tool_weights")
    os.environ["TOOL_WEIGHTS_ROOT"] = default_value
    # print(f"Environment variable TOOL_WEIGHTS_ROOT not set. Using default: {default_value}")
