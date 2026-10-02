FROM httpd:2.4.50

# Fix Debian Buster archived repositories
RUN sed -i 's/deb.debian.org/archive.debian.org/g' /etc/apt/sources.list && \
    sed -i 's|security.debian.org/debian-security|archive.debian.org/debian-security|g' /etc/apt/sources.list && \
    sed -i '/buster-updates/d' /etc/apt/sources.list && \
    echo 'Acquire::Check-Valid-Until "false";' > /etc/apt/apt.conf.d/99no-check-valid-until

# Install Python 3, PostgreSQL drivers, and dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-psycopg2 \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create application log directory
RUN mkdir -p /var/log/app && chmod 777 /var/log/app

# Copy custom Apache configuration
COPY apache.conf /usr/local/apache2/conf/httpd.conf

# Copy static assets and CGI scripts
COPY htdocs /usr/local/apache2/htdocs
COPY cgi-bin /usr/local/apache2/cgi-bin

# Make CGI scripts executable
RUN chmod -R 755 /usr/local/apache2/cgi-bin

EXPOSE 80
