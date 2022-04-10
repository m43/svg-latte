from datetime import datetime


class Object(object):
    pass


def get_str_formatted_time() -> str:
    return datetime.now().strftime('%Y.%m.%d_%H.%M.%S')
