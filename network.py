import socket
from urllib.parse import urlparse

def is_localhost(url):
    parsed = urlparse(url)

    # return None if it is no valid URL
    if not all([parsed.scheme, parsed.netloc]):
        return None

    if parsed.hostname in ['127.0.0.1', 'localhost']:
        return True

    hostname = socket.gethostname()
    if parsed.hostname == hostname:
        return True

    ip = socket.gethostbyname(hostname)
    if parsed.hostname == ip:
        return True

    return False
