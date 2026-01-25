"""
=============================================================================
AUTHENTICATION MODULE - XÁC THỰC VÀ QUẢN LÝ NGƯỜI DÙNG
=============================================================================
Mô tả:
    Module xử lý xác thực (authentication) và quản lý người dùng.
    Lưu trữ users trong file JSON đơn giản.

CÁC VAI TRÒ (ROLES):
    - admin: Quản trị viên - Full quyền
    - model_builder: Xây dựng mô hình - Không có admin settings
    - validator: Kiểm định viên - Chỉ xem
    - scorer: Người chấm điểm - Chỉ prediction

TÀI KHOẢN MẶC ĐỊNH (Development):
    | Username  | Password     | Role          |
    |-----------|--------------|---------------|
    | admin     | admin123     | admin         |
    | builder   | builder123   | model_builder |
    | validator | validator123 | validator     |
    | scorer    | scorer123    | scorer        |

Các hàm chính:
    - authenticate(): Xác thực username/password
    - get_user(): Lấy thông tin user theo username
    - get_all_users(): Lấy danh sách tất cả users
    - create_user(): Tạo user mới
    - update_user(): Cập nhật thông tin user
    - delete_user(): Xóa user
    - hash_password(): Hash mật khẩu bằng SHA-256
    - verify_password(): Xác minh mật khẩu

Bảo mật:
    - Mật khẩu được hash bằng SHA-256
    - Input được sanitize để loại bỏ ký tự Unicode ẩn (từ Unikey)
=============================================================================
"""

import json
import hashlib
import os
from dataclasses import dataclass
from typing import Optional, Dict, List
from pathlib import Path


# =============================================================================
# ĐỊNH NGHĨA VAI TRÒ (ROLE DEFINITIONS)
# =============================================================================
ROLES = ['admin', 'model_builder', 'validator', 'scorer']
ROLE_NAMES = {
    'admin': 'Quản trị viên',
    'model_builder': 'Xây dựng mô hình',
    'validator': 'Kiểm định viên', 
    'scorer': 'Người chấm điểm'
}


# =============================================================================
# USER MODEL CLASS
# =============================================================================

@dataclass
class User:
    """User model"""
    username: str
    role: str
    display_name: str
    password_hash: str = ""
    
    def to_dict(self) -> dict:
        return {
            'username': self.username,
            'role': self.role,
            'display_name': self.display_name,
            'password_hash': self.password_hash
        }
    
    @classmethod
    def from_dict(cls, username: str, data: dict) -> 'User':
        return cls(
            username=username,
            role=data.get('role', 'scorer'),
            display_name=data.get('display_name', username),
            password_hash=data.get('password_hash', '')
        )


def _get_users_file_path() -> Path:
    """Get path to users.json file"""
    return Path(__file__).parent / 'users.json'


def _load_users() -> Dict[str, dict]:
    """Load users from JSON file"""
    users_file = _get_users_file_path()
    if users_file.exists():
        try:
            with open(users_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {}


def _save_users(users: Dict[str, dict]) -> bool:
    """Save users to JSON file"""
    users_file = _get_users_file_path()
    try:
        with open(users_file, 'w', encoding='utf-8') as f:
            json.dump(users, f, ensure_ascii=False, indent=2)
        return True
    except IOError:
        return False


def hash_password(password: str) -> str:
    """Hash a password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()


def verify_password(password: str, password_hash: str) -> bool:
    """Verify a password against its hash"""
    return hash_password(password) == password_hash


def _sanitize_input(text: str) -> str:
    """
    Sanitize input to remove invisible characters that may be added by 
    Vietnamese input methods like Unikey.
    """
    import unicodedata
    if not text:
        return text
    # Strip whitespace
    text = text.strip()
    # Remove zero-width characters and other invisible Unicode
    invisible_chars = [
        '\u200b',  # Zero-width space
        '\u200c',  # Zero-width non-joiner
        '\u200d',  # Zero-width joiner
        '\ufeff',  # BOM / Zero-width no-break space
        '\u2060',  # Word joiner
        '\u00ad',  # Soft hyphen
    ]
    for char in invisible_chars:
        text = text.replace(char, '')
    # Normalize Unicode (NFC form)
    text = unicodedata.normalize('NFC', text)
    return text


def authenticate(username: str, password: str) -> Optional[User]:
    """
    Authenticate a user with username and password.
    Returns User object if successful, None otherwise.
    """
    # Sanitize inputs to handle Unikey/Vietnamese IME issues
    username = _sanitize_input(username)
    password = _sanitize_input(password)
    
    users = _load_users()
    
    if username not in users:
        return None
    
    user_data = users[username]
    if not verify_password(password, user_data.get('password_hash', '')):
        return None
    
    return User.from_dict(username, user_data)


def get_user(username: str) -> Optional[User]:
    """Get a user by username"""
    users = _load_users()
    if username in users:
        return User.from_dict(username, users[username])
    return None


def get_all_users() -> List[User]:
    """Get all users"""
    users = _load_users()
    return [User.from_dict(username, data) for username, data in users.items()]


def create_user(username: str, password: str, role: str, display_name: str = None) -> Optional[User]:
    """Create a new user"""
    if role not in ROLES:
        return None
    
    users = _load_users()
    if username in users:
        return None  # User already exists
    
    user = User(
        username=username,
        role=role,
        display_name=display_name or username,
        password_hash=hash_password(password)
    )
    
    users[username] = user.to_dict()
    del users[username]['username']  # Don't store username in value
    
    if _save_users(users):
        return user
    return None


def update_user(username: str, password: str = None, role: str = None, display_name: str = None) -> Optional[User]:
    """Update an existing user"""
    users = _load_users()
    if username not in users:
        return None
    
    user_data = users[username]
    
    if password:
        user_data['password_hash'] = hash_password(password)
    if role and role in ROLES:
        user_data['role'] = role
    if display_name:
        user_data['display_name'] = display_name
    
    users[username] = user_data
    
    if _save_users(users):
        return User.from_dict(username, user_data)
    return None


def delete_user(username: str) -> bool:
    """Delete a user"""
    users = _load_users()
    if username not in users:
        return False
    
    del users[username]
    return _save_users(users)


def init_default_users():
    """Initialize default users if users.json doesn't exist or is empty"""
    users = _load_users()
    
    if not users:
        default_users = {
            'admin': {
                'password_hash': hash_password('admin123'),
                'role': 'admin',
                'display_name': 'Quản trị viên'
            },
            'validator': {
                'password_hash': hash_password('validator123'),
                'role': 'validator',
                'display_name': 'Kiểm định viên'
            },
            'scorer': {
                'password_hash': hash_password('scorer123'),
                'role': 'scorer',
                'display_name': 'Nhân viên chấm điểm'
            }
        }
        _save_users(default_users)
        print("[OK] Default users created: admin, validator, scorer")


# Initialize default users on module import
init_default_users()
