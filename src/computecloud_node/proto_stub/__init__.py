"""Stub for gRPC imports — only used if --http is not set and grpcio is not installed.

The standalone node client uses HTTP transport by default (--http flag).
If a user tries gRPC mode without grpcio installed, this stub provides
a helpful error message.
"""

# noqa: F401 — this module exists only to prevent ImportError at module load.
