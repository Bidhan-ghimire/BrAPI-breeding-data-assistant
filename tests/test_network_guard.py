"""The conftest network guard is active for the tests in this folder (added 2026-10-14, Step 0).

Both checks are safe even if the guard were missing: 192.0.2.1 is a documentation-only
address (RFC 5737) that no computer answers, and the second check only looks at the code.
"""
import socket

import pytest


def test_a_connection_to_another_computer_is_refused(network_blocked_error):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        with pytest.raises(network_blocked_error):
            sock.connect(("192.0.2.1", 443))


def test_the_socket_functions_are_the_guarded_ones():
    assert socket.socket.connect.__name__ == "guarded_connect"
    assert socket.getaddrinfo.__name__ == "guarded_getaddrinfo"
