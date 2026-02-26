"""
File type validation using magic bytes — not client-supplied MIME headers.
"""
from fastapi import HTTPException

# Magic byte signatures for allowed formats
# Format: (offset, bytes_to_match)
IMAGE_SIGNATURES = {
    "image/jpeg": [(0, b'\xff\xd8\xff')],
    "image/png":  [(0, b'\x89PNG\r\n\x1a\n')],
    "image/bmp":  [(0, b'BM')],
    "image/webp": [(0, b'RIFF'), (8, b'WEBP')],  # both must match
}

VIDEO_SIGNATURES = {
    "video/mp4":        [(4, b'ftyp')],
    "video/quicktime":  [(4, b'ftyp'), (4, b'moov')],  # either matches
    "video/x-matroska": [(0, b'\x1a\x45\xdf\xa3')],
    "video/avi":        [(0, b'RIFF'), (8, b'AVI ')],
}

ALLOWED_IMAGE_TYPES = set(IMAGE_SIGNATURES.keys())
ALLOWED_VIDEO_TYPES = set(VIDEO_SIGNATURES.keys())


def _read_header(data: bytes, offset: int, length: int) -> bytes:
    """Safely read bytes from a specific offset."""
    if len(data) < offset + length:
        return b''
    return data[offset:offset + length]


def _check_signatures(header: bytes, signatures: list) -> bool:
    """
    Check if header matches ALL signature tuples in the list.
    For formats like WEBP/AVI that need two matches.
    """
    for offset, magic in signatures:
        chunk = _read_header(header, offset, len(magic))
        if chunk != magic:
            return False
    return True


def detect_image_type(header: bytes) -> str | None:
    """
    Detect image type from magic bytes.
    Returns MIME type string or None if unrecognised.
    """
    for mime_type, signatures in IMAGE_SIGNATURES.items():
        if _check_signatures(header, signatures):
            return mime_type
    return None


def detect_video_type(header: bytes) -> str | None:
    """
    Detect video type from magic bytes.
    MOV/MP4 share the ftyp box but older MOV files use moov/mdat/wide instead.
    """
    if len(header) < 12:
        return None

    # Read the box type at offset 4 (standard QuickTime/MPEG-4 container structure)
    box_type = _read_header(header, 4, 4)

    # MP4 and modern MOV — both use ftyp box
    if box_type == b'ftyp':
        return "video/mp4"

    # Older QuickTime MOV — uses moov, mdat, wide, free, or skip as first box
    if box_type in (b'moov', b'mdat', b'wide', b'free', b'skip', b'pnot'):
        return "video/quicktime"

    # MKV
    if _read_header(header, 0, 4) == b'\x1a\x45\xdf\xa3':
        return "video/x-matroska"

    # AVI
    if (_read_header(header, 0, 4) == b'RIFF' and
            _read_header(header, 8, 4) == b'AVI '):
        return "video/avi"

    return None


def validate_image_bytes(contents: bytes) -> str:
    """
    Validate image bytes using magic bytes.
    Returns detected MIME type.
    Raises HTTPException 400 if invalid or unsupported format.
    """
    if len(contents) < 12:
        raise HTTPException(status_code=400, detail="File too small to be a valid image")

    detected = detect_image_type(contents[:16])

    if detected is None:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "unsupported_format",
                "message": "File is not a recognised image format.",
                "supported_formats": ["JPEG", "PNG", "BMP", "WebP"],
            }
        )

    return detected


def validate_video_header(header: bytes, filename: str = "") -> str:
    """
    Validate video file header using magic bytes.
    Call this before writing the full file to disk.
    Returns detected MIME type.
    Raises HTTPException 400 if invalid or unsupported format.
    """
    if len(header) < 12:
        raise HTTPException(status_code=400, detail="File too small to be a valid video")

    detected = detect_video_type(header)

    if detected is None:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "unsupported_format",
                "message": "File is not a recognised video format.",
                "supported_formats": ["MP4", "MOV", "MKV", "AVI"],
                "hint": "Ensure the file is not corrupted and is a supported format.",
            }
        )

    return detected