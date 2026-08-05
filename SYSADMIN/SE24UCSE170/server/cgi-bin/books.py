#!/usr/bin/env python3
import cgi
import html
import sys
import os

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
        log_warn(f"Non-librarian access attempt to books page by user {current_user.get('username')}", username=current_user.get("username"))
        render_page("Access Denied", '<div class="alert alert-error">Access Denied: Only librarian accounts are permitted to manage books and locations.</div>', current_user=current_user)
        return

    form = cgi.FieldStorage()
    method = os.environ.get("REQUEST_METHOD", "GET").upper()
    action = form.getvalue("action", "")
    alert = None
    alert_type = "info"

    if method == "POST":
        if action == "add_book":
            title = form.getvalue("title", "").strip()
            author = form.getvalue("author", "").strip()
            isbn = form.getvalue("isbn", "").strip()
            genre = form.getvalue("genre", "").strip()
            location_shelf = form.getvalue("location_shelf", "").strip()
            location_section = form.getvalue("location_section", "").strip()

            if not title or not author or not location_shelf or not location_section:
                alert = "Title, Author, Shelf, and Section fields are required."
                alert_type = "error"
            else:
                query = """
                    INSERT INTO books (title, author, isbn, genre, location_shelf, location_section, status)
                    VALUES (%s, %s, %s, %s, %s, %s, 'available')
                    RETURNING id
                """
                res = execute_query(query, (title, author, isbn, genre, location_shelf, location_section), fetch_one=True, commit=True)
                audit_log(current_user["id"], current_user["username"], "ADD_BOOK", f"Added book '{title}' by {author} at {location_shelf}/{location_section}")
                log_info(f"Book added: {title} (ID: {res['id']})", username=current_user["username"])
                alert = f"Book '<strong>{html.escape(title)}</strong>' added successfully to inventory."
                alert_type = "success"

        elif action == "update_location":
            book_id = form.getvalue("book_id", "")
            location_shelf = form.getvalue("location_shelf", "").strip()
            location_section = form.getvalue("location_section", "").strip()

            if book_id and book_id.isdigit() and location_shelf and location_section:
                query = """
                    UPDATE books
                    SET location_shelf = %s, location_section = %s, updated_at = CURRENT_TIMESTAMP
                    WHERE id = %s
                """
                execute_query(query, (location_shelf, location_section, int(book_id)), commit=True)
                audit_log(current_user["id"], current_user["username"], "UPDATE_LOCATION", f"Updated book ID {book_id} location to {location_shelf}/{location_section}")
                alert = f"Location for Book ID #{book_id} updated to Shelf: '{html.escape(location_shelf)}', Section: '{html.escape(location_section)}'."
                alert_type = "success"

        elif action == "delete_book":
            book_id = form.getvalue("book_id", "")
            if book_id and book_id.isdigit():
                target_book = execute_query("SELECT title FROM books WHERE id = %s", (int(book_id),), fetch_one=True)
                if target_book:
                    execute_query("DELETE FROM books WHERE id = %s", (int(book_id),), commit=True)
                    audit_log(current_user["id"], current_user["username"], "DELETE_BOOK", f"Deleted book ID {book_id} ({target_book['title']})")
                    alert = f"Book '<strong>{html.escape(target_book['title'])}</strong>' removed from inventory."
                    alert_type = "success"

    # Search / List logic
    search = form.getvalue("q", "").strip()
    if search:
        query = """
            SELECT id, title, author, isbn, genre, location_shelf, location_section, status
            FROM books
            WHERE title ILIKE %s OR author ILIKE %s OR genre ILIKE %s OR location_shelf ILIKE %s OR location_section ILIKE %s
            ORDER BY title ASC
        """
        like_search = f"%{search}%"
        books = execute_query(query, (like_search, like_search, like_search, like_search, like_search))
    else:
        query = """
            SELECT id, title, author, isbn, genre, location_shelf, location_section, status
            FROM books
            ORDER BY title ASC
        """
        books = execute_query(query)

    search_val = html.escape(search)

    rows_html = ""
    if books:
        for b in books:
            b_id = b['id']
            title_esc = html.escape(b['title'])
            author_esc = html.escape(b['author'])
            shelf_esc = html.escape(b['location_shelf'])
            section_esc = html.escape(b['location_section'])
            status = b['status']
            badge_cls = "badge-available" if status == "available" else "badge-borrowed"

            rows_html += f'''
            <tr>
                <td><strong>#{b_id}</strong></td>
                <td><strong>{title_esc}</strong><br><small style="color:var(--text-muted);">{html.escape(b['isbn'] or '')}</small></td>
                <td>{author_esc}</td>
                <td>{html.escape(b['genre'] or '—')}</td>
                <td>
                    <form method="POST" action="/cgi-bin/books.py" style="display:flex; gap:0.4rem; align-items:center;">
                        <input type="hidden" name="action" value="update_location">
                        <input type="hidden" name="book_id" value="{b_id}">
                        <input type="text" name="location_shelf" value="{shelf_esc}" class="form-control" style="padding:0.25rem 0.4rem; font-size:0.8rem; width:80px;" placeholder="Shelf" required>
                        <input type="text" name="location_section" value="{section_esc}" class="form-control" style="padding:0.25rem 0.4rem; font-size:0.8rem; width:110px;" placeholder="Section" required>
                        <button type="submit" class="btn btn-secondary btn-sm" title="Save Location">Save</button>
                    </form>
                </td>
                <td><span class="badge {badge_cls}">{status.capitalize()}</span></td>
                <td>
                    <form method="POST" action="/cgi-bin/books.py" style="display:inline;" onsubmit="return confirm('Remove this book from inventory?');">
                        <input type="hidden" name="action" value="delete_book">
                        <input type="hidden" name="book_id" value="{b_id}">
                        <button type="submit" class="btn btn-danger btn-sm">Delete</button>
                    </form>
                </td>
            </tr>
            '''
    else:
        rows_html = '<tr><td colspan="7" style="text-align: center; color: var(--text-secondary); padding: 2rem;">No books registered in inventory.</td></tr>'

    body = f'''
    <div class="page-header">
        <div>
            <h1 class="page-title">Librarian Book & Location Management</h1>
            <p class="page-subtitle">Manage catalog inventory, update shelf & section locations</p>
        </div>
    </div>

    <div class="card">
        <div class="card-header">
            <h2 class="card-title">Add New Book to Inventory</h2>
        </div>
        <form method="POST" action="/cgi-bin/books.py">
            <input type="hidden" name="action" value="add_book">
            <div class="form-row">
                <div class="form-group">
                    <label class="form-label" for="title">Title *</label>
                    <input type="text" id="title" name="title" class="form-control" required placeholder="Book title">
                </div>
                <div class="form-group">
                    <label class="form-label" for="author">Author *</label>
                    <input type="text" id="author" name="author" class="form-control" required placeholder="Author full name">
                </div>
            </div>

            <div class="form-row">
                <div class="form-group">
                    <label class="form-label" for="isbn">ISBN</label>
                    <input type="text" id="isbn" name="isbn" class="form-control" placeholder="e.g. 978-0141183848">
                </div>
                <div class="form-group">
                    <label class="form-label" for="genre">Genre</label>
                    <input type="text" id="genre" name="genre" class="form-control" placeholder="e.g. Fiction / History">
                </div>
            </div>

            <div class="form-row">
                <div class="form-group">
                    <label class="form-label" for="location_shelf">Shelf Location *</label>
                    <input type="text" id="location_shelf" name="location_shelf" class="form-control" required placeholder="e.g. Shelf B-2">
                </div>
                <div class="form-group">
                    <label class="form-label" for="location_section">Section / Room *</label>
                    <input type="text" id="location_section" name="location_section" class="form-control" required placeholder="e.g. West Wing Classics">
                </div>
            </div>

            <div style="margin-top: 1rem;">
                <button type="submit" class="btn btn-primary">Add Book</button>
            </div>
        </form>
    </div>

    <div class="card">
        <div class="card-header" style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:1rem;">
            <h2 class="card-title">Book Inventory & Location Directory</h2>
        </div>

        <form method="GET" action="/cgi-bin/books.py" class="search-box">
            <input type="text" name="q" value="{search_val}" placeholder="Filter books by title, author, shelf, or section..." class="form-control">
            <button type="submit" class="btn btn-primary">Filter</button>
            {f'<a href="/cgi-bin/books.py" class="btn btn-secondary">Clear</a>' if search else ''}
        </form>

        <div class="table-responsive">
            <table class="data-table">
                <thead>
                    <tr>
                        <th>ID</th>
                        <th>Title</th>
                        <th>Author</th>
                        <th>Genre</th>
                        <th>Location (Shelf & Section)</th>
                        <th>Status</th>
                        <th>Action</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
        </div>
    </div>
    '''

    render_page("Book Inventory", body, current_user=current_user, alert=alert, alert_type=alert_type)

if __name__ == "__main__":
    main()
