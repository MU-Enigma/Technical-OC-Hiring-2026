#!/usr/bin/env python3
import cgi
import html
import sys
import os

# Include local package path if needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lib.db import execute_query
from lib.auth import get_current_user
from lib.template import render_page
from lib.logger import log_info

def main():
    form = cgi.FieldStorage()
    search = form.getvalue("q", "").strip()

    current_user = get_current_user()

    if search:
        query = """
            SELECT title, author, isbn, genre, location_shelf, location_section, status
            FROM books
            WHERE title ILIKE %s OR author ILIKE %s OR genre ILIKE %s OR location_shelf ILIKE %s OR location_section ILIKE %s
            ORDER BY title ASC
        """
        like_search = f"%{search}%"
        books = execute_query(query, (like_search, like_search, like_search, like_search, like_search))
    else:
        query = """
            SELECT title, author, isbn, genre, location_shelf, location_section, status
            FROM books
            ORDER BY title ASC
        """
        books = execute_query(query)

    search_val = html.escape(search)

    rows_html = ""
    if books:
        for b in books:
            status = b['status']
            badge_cls = "badge-available" if status == "available" else "badge-borrowed"
            rows_html += f'''
            <tr>
                <td><strong>{html.escape(b['title'])}</strong></td>
                <td>{html.escape(b['author'])}</td>
                <td>{html.escape(b['genre'] or '—')}</td>
                <td><span class="location-tag">{html.escape(b['location_shelf'])} ({html.escape(b['location_section'])})</span></td>
                <td><span class="badge {badge_cls}">{status.capitalize()}</span></td>
            </tr>
            '''
    else:
        rows_html = '<tr><td colspan="5" style="text-align: center; color: var(--text-secondary); padding: 2rem;">No books found matching your query.</td></tr>'

    body = f'''
    <div class="page-header">
        <div>
            <h1 class="page-title">Catalog Search</h1>
            <p class="page-subtitle">Browse books and physical locations at The Worm Library</p>
        </div>
    </div>

    <div class="card">
        <form method="GET" action="/cgi-bin/index.py" class="search-box">
            <input type="text" name="q" value="{search_val}" placeholder="Search by title, author, genre, or shelf location..." class="form-control">
            <button type="submit" class="btn btn-primary">Search Catalog</button>
            {f'<a href="/cgi-bin/index.py" class="btn btn-secondary">Clear</a>' if search else ''}
        </form>

        <div class="table-responsive">
            <table class="data-table">
                <thead>
                    <tr>
                        <th>Title</th>
                        <th>Author</th>
                        <th>Genre</th>
                        <th>Location (Shelf & Section)</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
        </div>
    </div>
    '''

    render_page("Catalog", body, current_user=current_user)

if __name__ == "__main__":
    main()
