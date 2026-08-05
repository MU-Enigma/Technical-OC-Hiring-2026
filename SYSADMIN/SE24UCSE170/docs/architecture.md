# System Architecture

## Overview
The Worm Library operates on a completely containerized architecture to ensure isolation. 

## Component Breakdown
1. **Client:** Librarian accessing the system via a standard web browser on the local network.
2. **Web Server:** Docker container running Apache 2.4.50. Handles HTTP requests and executes Python CGI scripts.
3. **Database:** Docker container running PostgreSQL. Stores library inventory and audit logs.
4. **Attacker:** External machine sending malicious HTTP requests to exploit CVE-2021-42013.

## Architecture Diagram

```mermaid
graph TD
    %% Normal Flow
    Client[Librarian Browser] -->|HTTP| Apache[Apache 2.4.50 Container]
    Apache -->|Executes| PythonCGI[Python CGI Scripts]
    PythonCGI -->|Reads/Writes| Postgres[(PostgreSQL Container)]
    
    %% Attack Flow
    Attacker[Attacker Machine] -.->|Malicious Path Traversal| Apache
    Apache -.->|Spawns| ReverseShell[Reverse Shell as daemon]
    ReverseShell -.->|Connects Back| Attacker
```
