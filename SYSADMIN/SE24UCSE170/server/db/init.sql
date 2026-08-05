-- The Worm Library Database Schema

CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(64) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(20) NOT NULL CHECK (role IN ('admin', 'librarian')),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS books (
    id SERIAL PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    author VARCHAR(255) NOT NULL,
    isbn VARCHAR(20),
    genre VARCHAR(100),
    location_shelf VARCHAR(50) NOT NULL,
    location_section VARCHAR(50) NOT NULL,
    status VARCHAR(20) DEFAULT 'available' CHECK (status IN ('available', 'borrowed', 'maintenance')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS borrow_records (
    id SERIAL PRIMARY KEY,
    book_id INT NOT NULL REFERENCES books(id) ON DELETE CASCADE,
    borrower_name VARCHAR(255) NOT NULL,
    borrower_contact VARCHAR(100) NOT NULL,
    borrower_email VARCHAR(255),
    borrow_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    due_date DATE NOT NULL,
    return_date TIMESTAMP WITH TIME ZONE,
    status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'returned')),
    created_by INT REFERENCES users(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS sessions (
    session_id VARCHAR(64) PRIMARY KEY,
    user_id INT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL
);

CREATE TABLE IF NOT EXISTS audit_logs (
    id SERIAL PRIMARY KEY,
    user_id INT REFERENCES users(id) ON DELETE SET NULL,
    username VARCHAR(64),
    action VARCHAR(100) NOT NULL,
    details TEXT,
    ip_address VARCHAR(45),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexing for fast search
CREATE INDEX IF NOT EXISTS idx_books_title ON books(title);
CREATE INDEX IF NOT EXISTS idx_books_author ON books(author);
CREATE INDEX IF NOT EXISTS idx_books_status ON books(status);
CREATE INDEX IF NOT EXISTS idx_borrow_records_status ON borrow_records(status);
CREATE INDEX IF NOT EXISTS idx_sessions_expires ON sessions(expires_at);

-- Seed initial admin account (password: WormAdmin2026!SecurePass)
-- Format: salt$hash where salt is w0rm_s4lt_v1
INSERT INTO users (username, password_hash, role, is_active)
VALUES (
    'admin',
    'w0rm_s4lt_v1$03907cd9b4159fbc0bfb80c7b781a04453f6c47bc48ade3d365512e4405fb3c2',
    'admin',
    TRUE
) ON CONFLICT (username) DO NOTHING;

-- Seed initial sample librarian account (password: LibrarianPass2026!)
INSERT INTO users (username, password_hash, role, is_active)
VALUES (
    'librarian1',
    'w0rm_s4lt_v1$81c68374febbde640d37a9cb8591a195b2668efad9d04ab430470e22290cec5a',
    'librarian',
    TRUE
) ON CONFLICT (username) DO NOTHING;

-- Seed initial books catalog for The Worm Library
INSERT INTO books (title, author, isbn, genre, location_shelf, location_section, status) VALUES
('The Book of Sand', 'Jorge Luis Borges', '978-0141183848', 'Fiction / Fantasy', 'Shelf A-1', 'Main Hall - Rare Books', 'available'),
('Dune', 'Frank Herbert', '978-0441172719', 'Science Fiction', 'Shelf B-3', 'West Wing - Sci-Fi', 'available'),
('Fahrenheit 451', 'Ray Bradbury', '978-1451673319', 'Dystopian Fiction', 'Shelf B-4', 'West Wing - Classics', 'available'),
('The Name of the Rose', 'Umberto Eco', '978-0156001311', 'Historical Fiction', 'Shelf C-2', 'East Reading Room', 'available'),
('Invisible Cities', 'Italo Calvino', '978-0156453806', 'Philosophical Fiction', 'Shelf A-2', 'Main Hall - Poetry & Prose', 'available');
