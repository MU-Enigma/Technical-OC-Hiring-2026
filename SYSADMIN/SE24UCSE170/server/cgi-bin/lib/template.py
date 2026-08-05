import sys

def render_page(title, body_content, current_user=None, alert=None, alert_type="info", extra_headers=None):
    """Render full HTML dynamic CGI page with clean modern aesthetic."""
    headers = ["Content-Type: text/html; charset=utf-8"]
    if extra_headers:
        if isinstance(extra_headers, list):
            headers.extend(extra_headers)
        else:
            headers.append(extra_headers)

    for h in headers:
        print(h)
    print()  # Empty line separating HTTP headers from body

    # Navigation bar items based on user role
    nav_items = []
    if current_user:
        role = current_user.get("role")
        username = current_user.get("username")
        if role == "admin":
            nav_items.append('<a href="/cgi-bin/admin.py" class="nav-link active">Manage Librarians</a>')
        elif role == "librarian":
            nav_items.append('<a href="/cgi-bin/books.py" class="nav-link">Book Inventory & Locations</a>')
            nav_items.append('<a href="/cgi-bin/borrow.py" class="nav-link">Borrowing & Returns</a>')

        user_badge = f'<span class="user-badge"><span class="dot"></span> {username} ({role.capitalize()})</span>'
        nav_items.append(user_badge)
        nav_items.append('<a href="/cgi-bin/logout.py" class="btn btn-outline-sm">Log Out</a>')
    else:
        nav_items.append('<a href="/cgi-bin/index.py" class="nav-link">Catalog</a>')
        nav_items.append('<a href="/cgi-bin/login.py" class="btn btn-primary-sm">Staff Login</a>')

    nav_html = "\n".join(nav_items)

    # Flash alert message block
    alert_html = ""
    if alert:
        alert_html = f'''
        <div class="alert alert-{alert_type}">
            <span class="alert-icon">ℹ️</span>
            <span>{alert}</span>
        </div>
        '''

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} | The Worm Library</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:ital,wght@0,600;0,700;1,400&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="/css/style.css">
</head>
<body>
    <header class="site-header">
        <div class="header-container">
            <a href="/cgi-bin/index.py" class="brand">
                <span class="brand-icon">📚</span>
                <span class="brand-name">The Worm Library</span>
            </a>
            <nav class="main-nav">
                {nav_html}
            </nav>
        </div>
    </header>

    <main class="main-content">
        <div class="container">
            {alert_html}
            {body_content}
        </div>
    </main>

    <footer class="site-footer">
        <div class="container footer-content">
            <p>&copy; 2026 <strong>The Worm Library</strong>. Minimalist Library System.</p>
            <p class="server-tag">Apache 2.4.50 • PostgreSQL • CGI Engine</p>
        </div>
    </footer>
</body>
</html>
'''
    print(html)
