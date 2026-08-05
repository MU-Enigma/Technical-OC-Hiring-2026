# Detection and Patching Guide

## 1. Incident Detection (Logging)
We need to verify the attack footprint in the server logs.

**Log Location:** `./volumes/apache_logs/access_log`

**Command used to analyze the logs:**
```bash
tail -n 50 ./volumes/apache_logs/access_log
cat ./volumes/apache_logs/access_log | grep -E \
    "/bin/sh|/bin/bash|/etc/passwd"
```

**What we found:**
* A weird request like:
    - `GET /cgi-bin/.%%32%65/.%%32%65/.%%32%65/.%%32%65/.%%32%65/etc/passwd`

    - or `xcmX /cgi-bin/.%%32%65/.%%32%65/.%%32%65/.%%32%65/.%%32%65/bin/sh`

    - or `hTv /cgi-bin/.%%32%65/.%%32%65/.%%32%65/.%%32%65/.%%32%65/bin/sh HTTP/1.1`

Since `/bin/sh` is used, we can assume a shell connection might have been initated.

---

## 2. Mitigation Strategies
There are multiple ways to patch this specific Apache vulnerability. Below are the primary methods used to secure the environment.

**Note:**
- Patched config files are in this folder with a `*.patched` suffix. You can either follow the instructions to edit them or remove the suffix and place them in the server directory. But make sure to follow the rest of the instructions if you use these.

### Option A: Configuration Fix (Workaround)
CVE-2021-42013 occurs by escaping the web document root to access and execute system files like `/bin/sh`. We can mitigate this by denying access to the server's root filesystem.

**Steps:**
1. Open the configuration file: `apache.conf`
2. Apply the following changes:
    ```apache
    # Old Configuration
    <Directory />
        AllowOverride none
        Require all granted
    </Directory>

    # New Configuration
    <Directory />
        AllowOverride none
        Require all denied
    </Directory>
    ```
3. Rebuild the Apache service with the new config:
    ```bash
    docker compose up -d --build web
    ```

### Option B: Version Upgrade (Permanent Fix)
CVE-2021-42013 only affects Apache 2.4.49 and Apache 2.4.50 and not earlier versions. Upgrading to a later version will fix it.

**Steps:**
1. Edit `Dockerfile` to change Apache version.
    ```bash
    # Old version
    FROM httpd:2.4.50

    # New version (anything >=2.4.51 works)
    FROM httpd:2.4.51
    ```
2. Remove the "Fix Debian Buster archived repositories" `RUN` command. Apache 2.4.51 docker image uses Debian 11 (Bullseye) instead of Debian 10 (Buster) like 2.4.50. That `RUN` command fixed broken repo links in Buster, which is not required in Bullseye.
    ```bash
    # REMOVE THIS BLOCK OF CODE

    # --- START ---
    RUN sed -i 's/deb.debian.org/archive.debian.org/g' /etc/apt/sources.list && \

    sed -i 's|security.debian.org/debian-security|archive.debian.org/debian-security|g' /etc/apt/sources.list && \

    sed -i '/buster-updates/d' /etc/apt/sources.list && \

    echo 'Acquire::Check-Valid-Until "false";' > /etc/apt/apt.conf.d/99no-check-valid-until 
    # --- END ---
    ```

3. Rebuild the web service.
    ```bash
    docker compose up -d --build web
    ```

---

## 3. Verification (Proof of Patch)
To prove the mitigation works, we replay the exact attack from the `attacker` environment.

**Execution:**
```bash
msfconsole
```

```bash
msf> search CVE-2021-42013
msf> use exploit/multi/http/apache_normalize_path_rce

msf> set RHOST <server_ip>
msf> set RPORT <apacher_port>
msf> set SSL false

msf> check
```

**Result:**
*   **Expected Output:**
    ```bash
    [*] Using auxiliary/scanner/http/apache_normalize_path as check
    [-] http://<server_ip>:<port> - The target is not vulnerable to CVE-2021-42013 (requires mod_cgi to be enabled).
    [*] Scanned 1 of 1 hosts (100% complete)
    [*] <server_ip>:<port> - The target is not exploitable.
    ```
*   **Status:** Exploit fails. The reverse shell is successfully blocked. CVE-2021-42013 is patched!