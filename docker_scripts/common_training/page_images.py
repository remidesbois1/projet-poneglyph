"""Source-page loading shared by OCR training exports (HTTP and private R2)."""

import io
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import quote, unquote, urlsplit

import requests
from PIL import Image, ImageOps
from tqdm import tqdm

def parse_r2_reference(reference: str, allowed_bucket: str):
    """Validate the canonical private-page reference used by the backend."""
    parts = urlsplit(reference)
    if (
        reference != reference.strip()
        or parts.scheme != "r2"
        or not allowed_bucket
        or parts.netloc != allowed_bucket
        or parts.query
        or parts.fragment
    ):
        raise ValueError("Invalid or unconfigured private page bucket")

    segments = parts.path.removeprefix("/").split("/")
    decoded = [unquote(segment, errors="strict") for segment in segments]
    for raw, value in zip(segments, decoded):
        if (
            not value
            or value in {".", ".."}
            or "/" in value
            or "\\" in value
            or any(ord(char) < 32 or ord(char) == 127 for char in value)
            or quote(value, safe="~!*'()-._") != raw
        ):
            raise ValueError("Invalid private page object key")
    return parts.netloc, "/".join(decoded)


def create_r2_client(download_workers=16, timeout_seconds=45):
    """Create a clock-skew-aware S3 client for Cloudflare R2 private pages."""
    required = (
        "R2_ENDPOINT",
        "R2_ACCESS_KEY_ID",
        "R2_SECRET_ACCESS_KEY",
        "R2_PAGES_BUCKET_NAME",
    )
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        raise RuntimeError(
            "Private page export requires: " + ", ".join(missing)
        )

    from datetime import datetime, timedelta, timezone
    from email.utils import parsedate_to_datetime

    import boto3
    from botocore.auth import AUTH_TYPE_MAPS, SIGV4_TIMESTAMP, S3SigV4Auth
    from botocore.config import Config as S3Config

    clock_offset = timedelta()

    class R2ClockAuth(S3SigV4Auth):
        def _modify_request_before_signing(self, request):
            request.context["timestamp"] = (
                datetime.now(timezone.utc) + clock_offset
            ).strftime(SIGV4_TIMESTAMP)
            super()._modify_request_before_signing(request)

    def clock_skew_retry(response=None, attempts=0, **_kwargs):
        nonlocal clock_offset
        parsed = response[1] if response else {}
        if (
            parsed.get("Error", {}).get("Code") == "RequestTimeTooSkewed"
            and attempts < 3
        ):
            date = (
                parsed.get("ResponseMetadata", {})
                .get("HTTPHeaders", {})
                .get("date")
            )
            if date:
                clock_offset = parsedate_to_datetime(date) - datetime.now(timezone.utc)
                return 0
        return None

    AUTH_TYPE_MAPS["poneglyph-r2-v4"] = R2ClockAuth
    client = boto3.client(
        "s3",
        endpoint_url=os.environ["R2_ENDPOINT"],
        region_name="auto",
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        config=S3Config(
            signature_version="poneglyph-r2-v4",
            retries={"max_attempts": 4, "mode": "standard"},
            max_pool_connections=max(download_workers, 8),
            connect_timeout=min(timeout_seconds, 30),
            read_timeout=timeout_seconds,
        ),
    )
    client.meta.events.register("needs-retry.s3.GetObject", clock_skew_retry)
    return client


def download_pages(pages, download_workers=16, timeout_seconds=45):
    print(f"Downloading {len(pages)} unique source pages...", flush=True)
    page_cache = {}
    has_private_r2 = any(
        str(data.get("url_image") or "").startswith("r2://")
        for data in pages.values()
    )
    r2_client = create_r2_client(download_workers, timeout_seconds) if has_private_r2 else None

    def download_page(page_id, url):
        if str(url).startswith("r2://"):
            from botocore.exceptions import BotoCoreError, ClientError

            bucket, key = parse_r2_reference(
                str(url), os.environ["R2_PAGES_BUCKET_NAME"]
            )
            try:
                response = r2_client.get_object(Bucket=bucket, Key=key)
                with response["Body"] as body:
                    content = body.read()
            except (BotoCoreError, ClientError) as exc:
                raise OSError(
                    f"private R2 read failed ({type(exc).__name__})"
                ) from None
        else:
            response = requests.get(url, timeout=timeout_seconds)
            response.raise_for_status()
            content = response.content

        try:
            with Image.open(io.BytesIO(content)) as img:
                page_img = ImageOps.exif_transpose(img).convert("RGB")
        except (OSError, ValueError) as exc:
            raise OSError(f"image decode failed ({type(exc).__name__})") from None
        return page_id, page_img

    with ThreadPoolExecutor(max_workers=download_workers) as executor:
        futures = {
            executor.submit(download_page, page_id, data["url_image"]): page_id
            for page_id, data in pages.items()
        }
        failures = []
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading pages"):
            page_id = futures[future]
            try:
                completed_page_id, image = future.result()
                page_cache[completed_page_id] = image
            except Exception as exc:
                failures.append((page_id, str(exc)))

    if failures:
        print(f"Failed source pages: {len(failures)}/{len(pages)}", flush=True)
        for page_id, message in failures[:20]:
            print(f"  page {page_id}: {message}", flush=True)
        if len(failures) > 20:
            print(f"  ... and {len(failures) - 20} more", flush=True)
        raise RuntimeError(
            "Source page download failed; refusing to create a partial dataset."
        )

    print(f"Cached pages: {len(page_cache)}/{len(pages)}", flush=True)
    return page_cache



