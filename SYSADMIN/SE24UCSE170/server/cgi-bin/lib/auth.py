import os
import secrets
import hashlib
from datetime import datetime, timedelta
import http.cookies
from lib.db import execute_query
from lib.logger import log_info, log_warn, log_error

SALT_PREFIX = "w0rm_s4lt_v1"
COOKIE_NAME = "worm_session"

def hash_password(password: str) -> str:
    """Hash password using salt prefix and SHA-256."""
    salted = f"{SALT_PREFIX}{password}".encode('utf-8')
    digest = hashlib.sha256(salted).hexdigest()
    return f"{SALT_PREFIX}${digest}"

def verify_password(password: str, stored_hash: str) -> bool:
    """Verify raw password against stored hash."""
    if not stored_hash or "$" not in stored_hash:
        return False
    salt, expected_digest = stored_hash.split("$", 1)
    salted = f"{salt}{password}".encode('utf-8')
    actual_digest = hashlib.sha256(salted).hexdigest()
    return secrets.compare_digest(actual_digest, expected_digest)

def get_session_cookie():
    """Extract worm_session cookie from request headers."""
    cookie_str = os.environ.get("HTTP_COOKIE", "")
    if not cookie_str:
        return None
    cookie = http.cookies.SimpleCookie()
    try:
        cookie.load(cookie_str)
        if COOKIE_NAME in cookie:
            return cookie[COOKIE_NAME].value
    except Exception:
        pass
    return None

def get_current_user():
    """Retrieve currently authenticated user record from session cookie."""
    session_id = get_session_cookie()
    if not session_id:
        return None

    query = """
        SELECT u.id, u.username, u.role, u.is_active
        FROM sessions s
        JOIN users u ON s.user_id = u.id
        WHERE s.session_id = %s
          AND s.expires_at > CURRENT_TIMESTAMP
          AND u.is_active = TRUE
    """
    user = execute_query(query, (session_id,), fetch_one=True)
    return user

def create_session(user_id: int, username: str) -> str:
    """Generate session ID, store in DB with 24h expiry, return set-cookie header string."""
    session_id = secrets.token_hex(32)
    query = """
        INSERT INTO sessions (session_id, user_id, expires_at)
        VALUES (%s, %s, CURRENT_TIMESTAMP + INTERVAL '24 hours')
    """
    execute_query(query, (session_id, user_id), commit=True)
    
    # Audit log
    audit_log(user_id, username, "USER_LOGIN", "User logged in successfully")
    log_info(f"Session created for user {username}", username=username)

    cookie = http.cookies.SimpleCookie()
    cookie[COOKIE_NAME] = session_id
    cookie[COOKIE_NAME]["path"] = "/"
    cookie[COOKIE_NAME]["httponly"] = True
    return cookie.output()

def destroy_session():
    """Delete session from DB and produce expired set-cookie header."""
    session_id = get_session_cookie()
    user = get_current_user()
    if session_id:
        execute_query("DELETE FROM sessions WHERE session_id = %s", (session_id,), commit=True)
        if user:
            audit_log(user["id"], user["username"], "USER_LOGOUT", "User logged out")
            log_info("User logged out", username=user["username"])

    cookie = http.cookies.SimpleCookie()
    cookie[COOKIE_NAME] = ""
    cookie[COOKIE_NAME]["path"] = "/"
    cookie[COOKIE_NAME]["expires"] = "Thu, 01 Jan 1970 00:00:00 GMT"
    return cookie.output()

def audit_log(user_id, username, action, details):
    """Record event in database audit table."""
    try:
        ip = os.environ.get("REMOTE_ADDR", "127.0.0.1")
        query = """
            INSERT INTO audit_logs (user_id, username, action, details, ip_address)
            VALUES (%s, %s, %s, %s, %s)
        """
        execute_query(query, (user_id, username, action, details, ip), commit=True)
    except Exception as e:
        log_error(f"Failed to record audit log: {str(e)}")
