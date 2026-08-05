#!/usr/bin/env python3
import cgi
import html
import sys
import os
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lib.db import execute_query
from lib.auth import get_current_user, audit_log
from lib.template import render_page
from lib.logger import log_info, log_warn, log_error

def main():
    current_user = get_current_user()
    if not current_user:
        print("Status: 302 Found")
        print("Location: /cgi-bin/login.py")
        print()
        return

    # Enforce strict Librarian Role check
    if current_user.get("role") != "librarian":
        log_warn(f"Non-librarian access attempt to borrowing page by user {current_user.get('username')}", username=current_user.get("username"))
        render_page("Access Denied", '<div class="alert alert-error">Access Denied: Only librarian accounts are permitted to manage book borrowings.</div>', current_user=current_user)
        return

    form = cgi.FieldStorage()
    method = os.environ.get("REQUEST_METHOD", "GET").upper()
    action = form.getvalue("action", "")
    alert = None
    alert_type = "info"

    if method == "POST":
        if action == "checkout_book":
            book_id = form.getvalue("book_id", "")
            borrower_name = form.getvalue("borrower_name", "").strip()
            borrower_contact = form.getvalue("borrower_contact", "").strip()
            borrower_email = form.getvalue("borrower_email", "").strip()
            due_days = form.getvalue("due_days", "14")

            if not book_id or not book_id.isdigit() or not borrower_name or not borrower_contact:
                alert = "Book selection, Borrower Name, and Contact Number are required."
                alert_type = "error"
            else:
                book = execute_query("SELECT id, title, status FROM books WHERE id = %s", (int(book_id),), fetch_one=True)
                if not book:
                    alert = "Selected book does not exist."
                    alert_type = "error"
                elif book["status"] != "available":
                    alert = f"Book '{html.escape(book['title'])}' is currently marked as {book['status']}."
                    alert_type = "error"
                else:
                    try:
                        days = int(due_days) if due_days.isdigit() else 14
                    except ValueError:
                        days = 14

                    # Insert borrow record and update book status
                    query_borrow = """
                        INSERT INTO borrow_records (book_id, borrower_name, borrower_contact, borrower_email, due_date, status, created_by)
                        VALUES (%s, %s, %s, %s, CURRENT_DATE + %s * INTERVAL '1 day', 'active', %s)
                    """
                    execute_query(query_borrow, (int(book_id), borrower_name, borrower_contact, borrower_email, days, current_user["id"]), commit=True)

                    query_update = "UPDATE books SET status = 'borrowed', updated_at = CURRENT_TIMESTAMP WHERE id = %s"
                    execute_query(query_update, (int(book_id),), commit=True)

                    audit_log(current_user["id"], current_user["username"], "CHECKOUT_BOOK", f"Borrowed '{book['title']}' to {borrower_name} ({borrower_contact})")
                    log_info(f"Book '{book['title']}' issued to borrower '{borrower_name}'", username=current_user["username"])
                    alert = f"Book '<strong>{html.escape(book['title'])}</strong>' has been checked out to <strong>{html.escape(borrower_name)}</strong>."
                    alert_type = "success"

        elif action == "return_book":
            record_id = form.getvalue("record_id", "")
            if record_id and record_id.isdigit():
                rec = execute_query("""
                    SELECT r.id, r.book_id, b.title, r.borrower_name
                    FROM borrow_records r
                    JOIN books b ON r.book_id = b.id
                    WHERE r.id = %s AND r.status = 'active'
                """, (int(record_id),), fetch_one=True)

                if rec:
                    # Update borrow record status to returned
                    execute_query("""
                        UPDATE borrow_records
                        SET status = 'returned', return_date = CURRENT_TIMESTAMP
                        WHERE id = %s
                    """, (int(record_id),), commit=True)

                    # Update book status to available
                    execute_query("UPDATE books SET status = 'available', updated_at = CURRENT_TIMESTAMP WHERE id = %s", (rec["book_id"],), commit=True)

                    audit_log(current_user["id"], current_user["username"], "RETURN_BOOK", f"Returned '{rec['title']}' from {rec['borrower_name']}")
                    log_info(f"Book '{rec['title']}' returned by borrower '{rec['borrower_name']}'", username=current_user["username"])
                    alert = f"Book '<strong>{html.escape(rec['title'])}</strong>' returned and location restored to available inventory."
                    alert_type = "success"

    # Fetch available books for checkout dropdown
    available_books = execute_query("""
        SELECT id, title, author, location_shelf, location_section
        FROM books
        WHERE status = 'available'
        ORDER BY title ASC
    """)

    book_options = ""
    if available_books:
        for b in available_books:
            book_options += f'<option value="{b["id"]}">{html.escape(b["title"])} by {html.escape(b["author"])} ({html.escape(b["location_shelf"])})</option>'
    else:
        book_options = '<option value="">No available books to check out</option>'

    # Fetch active borrow records
    active_borrows = execute_query("""
        SELECT r.id as record_id, b.id as book_id, b.title, b.author, b.location_shelf, b.location_section,
               r.borrower_name, r.borrower_contact, r.borrower_email, r.borrow_date, r.due_date
        FROM borrow_records r
        JOIN books b ON r.book_id = b.id
        WHERE r.status = 'active'
        ORDER BY r.due_date ASC
    """)

    active_rows_html = ""
    if active_borrows:
        for r in active_borrows:
            borrow_date_str = r['borrow_date'].strftime("%Y-%m-%d") if r['borrow_date'] else "—"
            due_date_str = r['due_date'].strftime("%Y-%m-%d") if r['due_date'] else "—"
            
            contact_info = html.escape(r['borrower_contact'])
            if r['borrower_email']:
                contact_info += f'<br><small style="color:var(--text-muted);">{html.escape(r["borrower_email"])}</small>'

            active_rows_html += f'''
            <tr>
                <td><strong>{html.escape(r['title'])}</strong><br><small style="color:var(--text-muted);">{html.escape(r['author'])}</small></td>
                <td><span class="location-tag">{html.escape(r['location_shelf'])} ({html.escape(r['location_section'])})</span></td>
                <td><strong>{html.escape(r['borrower_name'])}</strong></td>
                <td>{contact_info}</td>
                <td>{borrow_date_str}</td>
                <td><span class="badge badge-borrowed">Due: {due_date_str}</span></td>
                <td>
                    <form method="POST" action="/cgi-bin/borrow.py" style="display:inline;">
                        <input type="hidden" name="action" value="return_book">
                        <input type="hidden" name="record_id" value="{r['record_id']}">
                        <button type="submit" class="btn btn-primary-sm">Mark Returned</button>
                    </form>
                </td>
            </tr>
            '''
    else:
        active_rows_html = '<tr><td colspan="7" style="text-align: center; color: var(--text-secondary); padding: 1.5rem;">No active borrowings currently recorded.</td></tr>'

    body = f'''
    <div class="page-header">
        <div>
            <h1 class="page-title">Borrowing & Return Management</h1>
            <p class="page-subtitle">Mark books as borrowed with borrower information and manage active loans</p>
        </div>
    </div>

    <div class="card">
        <div class="card-header">
            <h2 class="card-title">Mark Book as Borrowed</h2>
        </div>
        <form method="POST" action="/cgi-bin/borrow.py">
            <input type="hidden" name="action" value="checkout_book">
            
            <div class="form-group">
                <label class="form-label" for="book_id">Select Book *</label>
                <select id="book_id" name="book_id" class="form-control" required>
                    {book_options}
                </select>
            </div>

            <div class="form-row">
                <div class="form-group">
                    <label class="form-label" for="borrower_name">Borrower Full Name *</label>
                    <input type="text" id="borrower_name" name="borrower_name" class="form-control" required placeholder="Full name of borrower">
                </div>
                <div class="form-group">
                    <label class="form-label" for="borrower_contact">Borrower Contact Phone *</label>
                    <input type="text" id="borrower_contact" name="borrower_contact" class="form-control" required placeholder="Phone number">
                </div>
            </div>

            <div class="form-row">
                <div class="form-group">
                    <label class="form-label" for="borrower_email">Borrower Email Address</label>
                    <input type="email" id="borrower_email" name="borrower_email" class="form-control" placeholder="Optional email address">
                </div>
                <div class="form-group">
                    <label class="form-label" for="due_days">Loan Duration (Days)</label>
                    <select id="due_days" name="due_days" class="form-control">
                        <option value="7">7 Days (1 Week)</option>
                        <option value="14" selected>14 Days (2 Weeks)</option>
                        <option value="30">30 Days (1 Month)</option>
                    </select>
                </div>
            </div>

            <div style="margin-top: 1rem;">
                <button type="submit" class="btn btn-primary" {"disabled" if not available_books else ""}>Check Out Book</button>
            </div>
        </form>
    </div>

    <div class="card">
        <div class="card-header">
            <h2 class="card-title">Currently Borrowed Books</h2>
        </div>
        <div class="table-responsive">
            <table class="data-table">
                <thead>
                    <tr>
                        <th>Book Title</th>
                        <th>Location</th>
                        <th>Borrower</th>
                        <th>Contact</th>
                        <th>Borrow Date</th>
                        <th>Due Date</th>
                        <th>Action</th>
                    </tr>
                </thead>
                <tbody>
                    {active_rows_html}
                </tbody>
            </table>
        </div>
    </div>
    '''

    render_page("Borrowing Management", body, current_user=current_user, alert=alert, alert_type=alert_type)

if __name__ == "__main__":
    main()
