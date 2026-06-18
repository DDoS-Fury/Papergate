"""Network-context classification shared by the synthetic generator and the
serving path, so train-time and serve-time agree on what counts as an *internal*
source network.

A SOURCE node is the client network a request originates from. Its registry key is
the client IPv4, **namespaced** ``src:<ip>`` so it can never collide with a device
key (``tpm:`` / ``ck:`` / ``ipdev:``) in the single, shared ``NodeRegistry`` slot
space. ``ip_is_internal`` produces the boolean fed into the source node's static
feature (``node_feat`` index 5): a private RFC1918 address is *internal*; anything
else (public, CGNAT 100.64/10, TEST-NET, loopback, link-local) is *external*.

This is a **model feature, not a policy gate**: per Zero Trust the network location
never grants authorization (that stays with OPA); the bit only conditions the
anomaly score so a never-seen access from an external network reads as more novel
than the same access from the corporate LAN.
"""

from __future__ import annotations

import ipaddress

SOURCE_PREFIX = "src:"

# Shared guest DEVICE node. The mirror of ``conf:guest``: when the experimental
# ``guest_device_fallback`` is on, every device that is NOT TPM-attested collapses
# onto this single anonymous, low-trust device node instead of carrying its own
# per-machine cookie (``ck:``) / weak-IP (``ipdev:``) identity.
GUEST_DEVICE = "dev:guest"


def to_guest_device(key_device):
    """Collapse any non-TPM device (``ck:`` / ``ipdev:`` / ``None``) onto the shared
    ``dev:guest`` node. A TPM-attested key (``tpm:<id>``) is returned unchanged."""
    if isinstance(key_device, str) and key_device.startswith("tpm:"):
        return key_device
    return GUEST_DEVICE

# Strict RFC1918 private ranges == "internal corporate network". Note we do NOT use
# ``ipaddress.is_private`` (it also returns True for loopback, link-local and CGNAT
# 100.64/10) — for the internal/external feature only these three count as internal.
_RFC1918 = [
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
]


def strip_source_prefix(key) -> str:
    """Return the bare IP from a (possibly ``src:``-namespaced) source key."""
    if isinstance(key, str) and key.startswith(SOURCE_PREFIX):
        return key[len(SOURCE_PREFIX):]
    return key


def ip_is_internal(key) -> bool:
    """True iff the source address is RFC1918 private (10/8, 172.16/12, 192.168/16).

    Accepts a bare IP or a ``src:<ip>`` registry key. CGNAT (100.64/10), public and
    TEST-NET addresses are external. Unparseable input is treated as external — a
    fail-safe default that never hands the 'internal' bit to a malformed source.
    """
    raw = strip_source_prefix(key)
    try:
        addr = ipaddress.ip_address(raw)
    except (ValueError, TypeError):
        return False
    return any(addr in net for net in _RFC1918)
