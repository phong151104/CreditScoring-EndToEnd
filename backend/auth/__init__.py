"""
Authentication Module - Backend
"""

from .auth import (
    User,
    authenticate,
    get_user,
    get_all_users,
    create_user,
    update_user,
    delete_user,
    hash_password,
    verify_password,
    init_default_users,
    ROLES,
    ROLE_NAMES
)

__all__ = [
    'User',
    'authenticate',
    'get_user',
    'get_all_users', 
    'create_user',
    'update_user',
    'delete_user',
    'hash_password',
    'verify_password',
    'init_default_users',
    'ROLES',
    'ROLE_NAMES'
]
