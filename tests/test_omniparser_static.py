"""Source-level regression checks for OmniParser (heavy GPU deps are not installed in CI)."""
import ast
import re

from conftest import ROOT

OMNI = ROOT / "src" / "omniparser"


def _read(rel):
    return (OMNI / rel).read_text(encoding="utf-8")


def test_utils_has_no_eager_paddleocr_import():
    tree = ast.parse(_read("util/utils.py"))
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names] + [getattr(node, "module", "") or ""]
            assert not any("paddleocr" in n for n in names), "paddleocr must be imported lazily"
    assert "def get_paddle_ocr" in _read("util/utils.py")


def test_no_ocr_text_does_not_zip_none():
    src = _read("util/utils.py")
    assert "ocr_bbox = None" not in src
    assert re.search(r"else:\s*\n\s*print\('no ocr bbox!!!'\)\s*\n\s*ocr_bbox = \[\]\s*\n\s*ocr_text = \[\]", src)


def test_device_from_config_is_honoured():
    src = _read("util/omniparser.py")
    assert "config.get('device'" in src
    assert "device = 'cuda' if torch.cuda.is_available() else 'cpu'" not in src


def test_server_not_reloading_and_binds_host():
    src = _read("server.py")
    assert "reload=True" not in src
    assert "host=config['host']" in src
    assert '"/health"' in src
