import os

try:
    import secrets_local
except ImportError:
    secrets_local = None

def _secret(env_name, local_attr):
    if os.environ.get(env_name):
        return os.environ[env_name]
    return getattr(secrets_local, local_attr, "") if secrets_local else ""

BASE_URL = "https://farmerchat.farmstack.co/mobile-app-stage"
API_KEY = _secret("FARMERCHAT_API_KEY", "FARMERCHAT_API_KEY")
LANGUAGE_ID = 1  # English
BUILD_VERSION = "v2"

# Langfuse (project: fc-agent-eagle, org: Digital Green Foundation)
LANGFUSE_HOST = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")
LANGFUSE_PUBLIC_KEY = _secret("LANGFUSE_PUBLIC_KEY", "LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = _secret("LANGFUSE_SECRET_KEY", "LANGFUSE_SECRET_KEY")

DEVICE_INFO_TEMPLATE = (
    "%7B%22manufacturer%22%3A%22Postman%22%2C%22brand%22%3A%22Postman%22%2C"
    "%22model%22%3A%22EvalRunner%22%2C%22app_version_code%22%3A%2293%22%2C"
    "%22app_version%22%3A%224.0.0%22%2C%22androidId%22%3A%22{device_id}%22%7D"
)

# Lat/long for Bihar (default)
DEFAULT_LAT = 25.0961
DEFAULT_LON = 85.3131

CSV_DIR = "/Users/sanyaamsingh/Downloads"
CSV_FILES = {
    "AMBG": f"{CSV_DIR}/English_Cases - AMBG.csv",
    "BRED": f"{CSV_DIR}/English_Cases - BRED.csv",
    "FLOW": f"{CSV_DIR}/English_Cases - FLOW.csv",
    "GROW": f"{CSV_DIR}/English_Cases - GROW.csv",
    "LVHL": f"{CSV_DIR}/English_Cases - LVHL.csv",
    "LVNU": f"{CSV_DIR}/English_Cases - LVNU.csv",
    "MRKT": f"{CSV_DIR}/English_Cases - MRKT.csv",
    "NUTR": f"{CSV_DIR}/English_Cases - NUTR.csv",
    "OOS":  f"{CSV_DIR}/English_Cases - OOS.csv",
    "PEST": f"{CSV_DIR}/English_Cases - PEST.csv",
    "SOIL": f"{CSV_DIR}/English_Cases - SOIL.csv",
    "WATR": f"{CSV_DIR}/English_Cases - WATR.csv",
    "WTHR": f"{CSV_DIR}/English_Cases - WTHR.csv",
}

RESULTS_DIR = "./results"
