"""Upload an algorithm container image to grand-challenge.org via gcapi.

Usage: python upload_to_gc.py <image.tar.gz> <algorithm-slug>
Token from env GC_TOKEN.
"""

import os
import sys

import gcapi


def main():
    path, slug = sys.argv[1], sys.argv[2]
    client = gcapi.Client(token=os.environ["GC_TOKEN"])

    algorithm = client.algorithms.detail(slug=slug)
    print("algorithm:", algorithm["api_url"])

    with open(path, "rb") as f:
        user_upload = client.uploads.upload_fileobj(
            fileobj=f, filename=os.path.basename(path))
    upload_url = (user_upload["api_url"]
                  if isinstance(user_upload, dict) else user_upload.api_url)
    print("uploaded:", upload_url)

    image = client.algorithm_images.create(
        algorithm=algorithm["api_url"], user_upload=upload_url)
    print("algorithm image created:",
          image.get("api_url", image) if isinstance(image, dict) else image)


if __name__ == "__main__":
    main()
