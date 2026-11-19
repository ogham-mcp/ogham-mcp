from ogham import okf


def test_okf_module_exposes_version_constant():
    assert okf.SUPPORTED_OKF_VERSION == "0.3"


def test_okf_module_exposes_public_api():
    assert callable(okf.export_okf_bundle)
    assert callable(okf.import_okf_bundle)
