# The Worm Library - Security Lifecycle Project

## Overview
This project demonstrates a complete server-client OS security lifecycle. The environment simulates "The Worm Library," a containerized library management system that allows staff to track books, update borrowed records, and process checkouts. 

* **Server:** A Docker containerized Apache 2.4.50 web server executing python `mod_cgi` scripts. 
* **Client:** Librarians accessing the server through standard web browsers on the local network. Communication is established via standard HTTP requests.


## Architecture
* **Web Server:** Apache 2.4.50
* **Application Layer:** Python CGI scripts
* **Database:** PostgreSQL container
* **Isolation:** The entire environment is containerized using Docker, allowing for quick deployment and limiting the damage of a potential breach.


## Journaling and Logging
The system implements a dual-layer logging approach to ensure complete visibility during an incident:
1. **Infrastructure Logging:** Built-in Apache logs (`access_log` and `error.log`) capture all raw HTTP traffic, connection attempts, and server errors.
2. **Application Auditing:** Custom Python CGI scripts record application-level actions (e.g., book updates, checkouts) into the PostgreSQL database for reliable audit trails.


## Security Lifecycle Demonstration
This project walks through the compromise and subsequent securing of the library server.

1. **The Attack (CVE-2021-42013):** The server runs a vulnerable version of Apache. We exploit a path traversal vulnerability to escape the web root and gain an unprivileged reverse shell as the `daemon` user.
2. **The Detection:** We utilize the Apache access and error logs to identify the malicious payload and trace the intrusion.
3. **The Mitigation:** We apply a configuration patch to enforce strict directory access controls, neutralizing the path traversal exploit.
4. **The Proof:** We rerun the exploit to demonstrate that the server now successfully blocks the attack.


## Project Structure
Please refer to the sub-directories for detailed documentation and reproduction steps:

* `server/` - Contains the server code, Docker configuration, and environment setup instructions.
* `attacker/` - Contains details on the CVE-2021-42013 vulnerability and instructions to execute the reverse shell exploit.
* `server_patch/` - Contains log analysis for detection, the mitigation strategy, proof of the successful patch and patched config files.
* `docs/` - Contains a video walkthrough and architecture notes.

---
**Disclaimer:** The Python CGI scripts used in this project were generated with the assistance of the Gemini Flash AI model as a helpful development resource.