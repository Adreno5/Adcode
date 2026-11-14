"""API-key encryption backed by the operating system keyring."""

from __future__ import annotations

from cryptography.fernet import Fernet

SERVICE = 'AdCodePythonCodingAgent'
KEY_NAME = 'fernet_key'


def _get_fernet() -> Fernet:
    import keyring
    key = keyring.get_password(SERVICE, KEY_NAME)
    if not key:
        key = Fernet.generate_key().decode()
        keyring.set_password(SERVICE, KEY_NAME, key)
    return Fernet(key.encode())


def encrypt_secret(plain: str) -> str:
    if not plain:
        return plain
    try:
        return _get_fernet().encrypt(plain.encode()).decode()
    except Exception:
        # A missing keyring backend must not destroy the stored value.
        return plain


def decrypt_secret(value: str) -> str:
    if not value:
        return value
    try:
        return _get_fernet().decrypt(value.encode()).decode()
    except Exception:
        return value
