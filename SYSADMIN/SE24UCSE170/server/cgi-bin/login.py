#!/usr/bin/env python3
import cgi
import html
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lib.db import execute_query
from lib.auth import verify_password, create_session, get_current_user, audit_log
from lib.template import render_page
from lib.logger import log_info, log_warn

def main():
    current_user = get_current_user()
    if current_user:
        role = current_user.get("role")
        target = "/cgi-bin/admin.py" if role == "admin" else "/cgi-bin/books.py"
        print("Status: 302 Found")
        print(f"Location: {target}")
        print()
        return

    method = os.environ.get("REQUEST_METHOD", "GET").upper()
    alert = None
    alert_type = "info"

    if method == "POST":
        form = cgi.FieldStorage()
        username = form.getvalue("username", "").strip()
        password = form.getvalue("password", "")

        if not username or not password:
            alert = "Please enter both username and password."
            alert_type = "error"
        else:
            query = "SELECT id, username, password_hash, role, is_active FROM users WHERE username = %s"
            user = execute_query(query, (username,), fetch_one=True)

            if user and user.get("is_active") and verify_password(password, user["password_hash"]):
                cookie_header = create_session(user["id"], user["username"])
                target = "/cgi-bin/admin.py" if user["role"] == "admin" else "/cgi-bin/books.py"
                print("Status: 302 Found")
                print(cookie_header)
                print(f"Location: {target}")
                print()
                return
            else:
                log_warn(f"Failed login attempt for username: {username}", username=username)
                audit_log(None, username, "FAILED_LOGIN", "Invalid credentials or inactive account")
                alert = "Invalid username or password, or account disabled."
                alert_type = "error"

    body = f'''
    <div class="auth-container">
        <div class="card">
            <div class="card-header" style="text-align: center;">
                <h2 style="font-family: 'Playfair Display', serif; font-size: 1.6rem; color: var(--brand-primary);">The Worm Library</h2>
                <p style="color: var(--text-secondary); font-size: 0.9rem; margin-top: 0.25rem;">Staff & Administrator Authentication</p>
            </div>

            <form method="POST" action="/cgi-bin/login.py">
                <div class="form-group">
                    <label class="form-label" for="username">Username</label>
                    <input type="text" id="username" name="username" class="form-control" required autofocus placeholder="Enter your staff username">
                </div>

                <div class="form-group">
                    <label class="form-label" for="password">Password</label>
                    <input type="password" id="password" name="password" class="form-control" required placeholder="Enter password">
                </div>

                <div style="margin-top: 1.5rem;">
                    <button type="submit" class="btn btn-primary" style="width: 100%;">Sign In</button>
                </div>
            </form>
        </div>
    </div>
    '''

    render_page("Staff Login", body, current_user=None, alert=alert, alert_type=alert_type)

if __name__ == "__main__":
    main()
