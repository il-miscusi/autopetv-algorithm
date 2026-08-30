"""Upload an algorithm container image to grand-challenge.org via gcapi.

Usage: python upload_to_gc.py <image.tar.gz> <algorithm-slug>
Token from env GC_TOKEN.
"""

import os
import sys

import gcapi


def field(obj, name):
    if isinstance(obj, dict):
        return obj[name]
    return getattr(obj, name)


def main():
    path, slug = sys.argv[1], sys.argv[2]
    client = gcapi.Client(token=os.environ["GC_TOKEN"])

    algorithm = client.algorithms.detail(slug=slug)
    alg_url = field(algorithm, "api_url")
    print("algorithm:", alg_url)

    with open(path, "rb") as f:
        user_upload = client.uploads.upload_fileobj(
            fileobj=f, filename=os.path.basename(path))
    upload_url = field(user_upload, "api_url")
    print("uploaded:", upload_url)

    image = client(method="POST", path="algorithms/images/",
                   json={"algorithm": alg_url, "user_upload": upload_url})
    print("algorithm image created:", image)


if __name__ == "__main__":
    main()
