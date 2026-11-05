import pytest

from postgresql.frame_repository import split_blob_url


@pytest.mark.parametrize("url,expected", [
    ("https://acct.blob.core.windows.net/frames/vid/vid_0000000003.jpg", ["frames", "vid", "vid_0000000003.jpg"]),
    ("http://127.0.0.1:10000/devstoreaccount1/frames/vid/vid_0000000003.jpg", ["frames", "vid", "vid_0000000003.jpg"]),
    ("http://azurite:10000/devstoreaccount1/frames/v.jpg", ["frames", "v.jpg"]),
])
def test_split_blob_url_handles_azure_and_azurite(url, expected):
    assert split_blob_url(url) == expected
