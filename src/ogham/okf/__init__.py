"""Open Knowledge Format (OKF) v0.3 support for Ogham.

OKF spec: https://github.com/open-knowledge-format/open-knowledge-format
Bundles written by earlier Ogham releases (OKF v0.1) still import.
"""

from ogham.okf.bundle import export_okf_bundle, import_okf_bundle

SUPPORTED_OKF_VERSION = "0.3"

__all__ = ["SUPPORTED_OKF_VERSION", "export_okf_bundle", "import_okf_bundle"]
