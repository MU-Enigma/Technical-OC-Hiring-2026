# The Worm Library - Library Management System

A simple library web application for running on Apache HTTP Server 2.4.50 with CGI scripts.

---

## Application Details

- Uses PostgreSQL to store data.
- Librarian accounts can manage books catalog and update borrow records. An admin account which can manage librarian accounts.
- All logs and data are stored inside `./volumes/`.

---

## Requirements

Make sure you have the following installed on your machine:
- Docker (v20.10+)
- Docker Compose (v2.0+)
- Git

---

## Setup & Installation Instructions

### Step 1: Clone the Repository
Clone the repository and open the server folder:
```bash
cd server
```

### Step 2: Environment Variables
Copy the template environment file to `.env`:
```bash
cp .env.example .env
```
*(Open `.env` and adjust passwords, database credentials, or application secret keys as needed).*

### Step 3: Build and Launch Containers
Start the Docker Compose services in detached mode:
```bash
docker compose up --build -d
```

### Step 4: Verify
Check that both `worm_library_db` (PostgreSQL) and `worm_library_web` (Apache) are up and healthy:
```bash
docker compose ps
```

Access the application in your web browser:
**`http://localhost:8080`** *(or custom port defined in `.env`)*

---

## Usage Instructions

### Initial Login Credentials

| Role | Username | Default Password | Initial Access |
| :--- | :--- | :--- | :--- |
| **Administrator** | `admin` | `WormAdmin2026!SecurePass` | `/cgi-bin/admin.py` |
| **Librarian** | `librarian1` | `LibrarianPass2026!` | `/cgi-bin/books.py` |

---

### Public Guest / Catalog Search
- Navigate to `http://localhost:8080/cgi-bin/index.py`.
- Search books and check their availability.

---

### Administrator Tasks (`/cgi-bin/admin.py`)
1. Log in as `admin`.
2. Create Librarian Accounts
3. View registered librarians, activate or disable librarian access.

---

### Librarian Tasks (`/cgi-bin/books.py` & `/cgi-bin/borrow.py`)
1. Log in as a librarian (e.g., `librarian1`).
2. Book Inventory & Location Management (`/cgi-bin/books.py`)
3. Borrowing and Checkout (`/cgi-bin/borrow.py`)
4. Processing Returns (`/cgi-bin/borrow.py`)

---

## Directory Structure

```
server/
├── .env                  # .env file
├── .env.example          # .env file template
├── .gitignore            # Git exclusion rules
├── Dockerfile            # Apache build manifest
├── README.md             # Documentation & setup guide
├── apache.conf           # Apache configuration
├── docker-compose.yml    # Service orchestration (web & db)
├── db/
│   └── init.sql          # PostgreSQL schema and initial seed data
├── htdocs/
│   ├── index.html        # Fallback landing redirect
│   └── css/
│       └── style.css     # Clean minimalist design system stylesheet
├── cgi-bin/
│   ├── index.py          # Public catalog search page
│   ├── login.py          # Staff authentication handler
│   ├── logout.py         # Session destruction handler
│   ├── admin.py          # Admin portal (Librarian account management)
│   ├── books.py          # Librarian inventory & location management
│   ├── borrow.py         # Librarian borrowing & return management
│   └── lib/
│       ├── auth.py       # Authentication, session cookies, audit log
│       ├── db.py         # Database connection pool & query engine
│       ├── logger.py     # Centralized structured logging module
│       └── template.py   # Page rendering engine and layout decorator
└── volumes/              # Project-local persistent storage
    ├── apache_logs/      # Apache server logs (access_log, error_log)
    ├── app_logs/         # Application logs (library.log)
    └── db_data/          # PostgreSQL database data files
```

---

## Logs & Diagnostics

- **Apache Access & Error Logs**:
  ```bash
  tail -f volumes/apache_logs/access_log
  tail -f volumes/apache_logs/error_log
  ```
- **Application Activity Logs**:
  ```bash
  tail -f volumes/app_logs/library.log
  ```
- **Apache Version Verification**:
  ```bash
  docker compose exec web httpd -v
  ```

---

## Stopping Services

To stop containers while retaining all database data and logs:
```bash
docker compose down
```

To stop containers and reset persistent volumes:
```bash
docker compose down -v
```
