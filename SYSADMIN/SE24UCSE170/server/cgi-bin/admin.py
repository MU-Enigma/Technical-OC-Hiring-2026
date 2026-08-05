#!/usr/bin/env python3
import cgi
import html
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lib.db import execute_query
from lib.auth import get_current_user, hash_password, audit_log
from lib.template import render_page
from lib.logger import log_info, log_warn, log_error

def main():
    current_user = get_current_user()
    if not current_user:
        print("Status: 302 Found")
        print("Location: /cgi-bin/login.py")
        print()
        return

    # Enforce strict Admin Role Isolation
    if current_user.get("role") != "admin":
        log_warn(f"Unauthorized admin page access attempt by user {current_user.get('username')}", username=current_user.get("username"))
        render_page("Access Denied", '<div class="alert alert-error">Access Denied: Admin privileges required. Admin accounts may only manage librarian accounts.</div>', current_user=current_user)
        return

    method = os.environ.get("REQUEST_METHOD", "GET").upper()
    alert = None
    alert_type = "info"

    if method == "POST":
        form = cgi.FieldStorage()
        action = form.getvalue("action", "")

        if action == "add_librarian":
            new_username = form.getvalue("username", "").strip()
            new_password = form.getvalue("password", "").strip()

            if not new_username or not new_password:
                alert = "Username and password are required to create a librarian account."
                alert_type = "error"
            elif len(new_password) < 6:
                alert = "Password must be at least 6 characters long."
                alert_type = "error"
            else:
                existing = execute_query("SELECT id FROM users WHERE username = %s", (new_username,), fetch_one=True)
                if existing:
                    alert = f"User '{html.escape(new_username)}' already exists."
                    alert_type = "error"
                else:
                    pw_hash = hash_password(new_password)
                    query = """
                        INSERT INTO users (username, password_hash, role, is_active)
                        VALUES (%s, %s, 'librarian', TRUE)
                    """
                    execute_query(query, (new_username, pw_hash), commit=True)
                    audit_log(current_user["id"], current_user["username"], "ADD_LIBRARIAN", f"Created librarian account: {new_username}")
                    log_info(f"Admin created new librarian account: {new_username}", username=current_user["username"])
                    alert = f"Librarian account '{html.escape(new_username)}' created successfully."
                    alert_type = "success"

        elif action == "toggle_status":
            user_id = form.getvalue("user_id", "")
            if user_id and user_id.isdigit():
                target_user = execute_query("SELECT id, username, is_active, role FROM users WHERE id = %s", (int(user_id),), fetch_one=True)
                if target_user and target_user["role"] == "librarian":
                    new_status = not target_user["is_active"]
                    execute_query("UPDATE users SET is_active = %s WHERE id = %s", (new_status, int(user_id)), commit=True)
                    status_str = "activated" if new_status else "disabled"
                    audit_log(current_user["id"], current_user["username"], "TOGGLE_LIBRARIAN", f"Set librarian {target_user['username']} active={new_status}")
                    alert = f"Librarian '{html.escape(target_user['username'])}' account has been {status_str}."
                    alert_type = "success"

    # Fetch all librarian accounts
    librarians = execute_query("""
        SELECT id, username, role, is_active, created_at
        FROM users
        WHERE role = 'librarian'
        ORDER BY created_at DESC
    """)

    rows_html = ""
    if librarians:
        for lib in librarians:
            status_badge = '<span class="badge badge-active">Active</span>' if lib['is_active'] else '<span class="badge badge-disabled">Disabled</span>'
            btn_text = "Disable Account" if lib['is_active'] else "Enable Account"
            btn_cls = "btn-danger" if lib['is_active'] else "btn-secondary"
            created_str = lib['created_at'].strftime("%Y-%m-%d %H:%M") if lib['created_at'] else "—"

            rows_html += f'''
            <tr>
                <td><strong>{html.escape(lib['username'])}</strong></td>
                <td><span class="badge badge-librarian">Librarian</span></td>
                <td>{status_badge}</td>
                <td>{created_str}</td>
                <td>
                    <form method="POST" action="/cgi-bin/admin.py" style="display:inline;">
                        <input type="hidden" name="action" value="toggle_status">
                        <input type="hidden" name="user_id" value="{lib['id']}">
                        <button type="submit" class="btn {btn_cls} btn-sm">{btn_text}</button>
                    </form>
                </td>
            </tr>
            '''
    else:
        rows_html = '<tr><td colspan="5" style="text-align: center; color: var(--text-secondary); padding: 1.5rem;">No librarian accounts created yet.</td></tr>'

    body = f'''
    <div class="page-header">
        <div>
            <h1 class="page-title">Admin Dashboard</h1>
            <p class="page-subtitle">Librarian Account Management Portal</p>
        </div>
    </div>

    <div class="card">
        <div class="card-header">
            <h2 class="card-title">Add New Librarian Account</h2>
        </div>
        <form method="POST" action="/cgi-bin/admin.py">
            <input type="hidden" name="action" value="add_librarian">
            <div class="form-row">
                <div class="form-group">
                    <label class="form-label" for="username">Librarian Username</label>
                    <input type="text" id="username" name="username" class="form-control" required placeholder="e.g. jsmith_librarian">
                </div>
                <div class="form-group">
                    <label class="form-label" for="password">Initial Password</label>
                    <input type="password" id="password" name="password" class="form-control" required placeholder="Minimum 6 characters">
                </div>
            </div>
            <div style="margin-top: 1rem;">
                <button type="submit" class="btn btn-primary">Create Librarian Account</button>
            </div>
        </form>
    </div>

    <div class="card">
        <div class="card-header">
            <h2 class="card-title">Registered Librarians</h2>
        </div>
        <div class="table-responsive">
            <table class="data-table">
                <thead>
                    <tr>
                        <th>Username</th>
                        <th>Role</th>
                        <th>Status</th>
                        <th>Account Created</th>
                        <th>Management Actions</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
        </div>
    </div>
    '''

    render_page("Admin Dashboard", body, current_user=current_user, alert=alert, alert_type=alert_type)

if __name__ == "__main__":
    main()
