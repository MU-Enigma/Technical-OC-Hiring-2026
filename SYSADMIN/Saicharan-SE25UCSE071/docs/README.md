# AssetVault: Server-Client OS Security Lifecycle

A self-built, deliberately vulnerable internal file-storage service, used to
demonstrate a complete security lifecycle: build, log, attack, patch, prove.

## 1. Environment & Workflow

**Topology:** Two Ubuntu 24.04 VMs on a fully isolated VMware internal
network (`10.0.1.0/24`, no host adapter, no DHCP server, no route to the
internet during the attack/proof phases).

| Role | IP | Purpose |
|---|---|---|
| Server | 10.0.1.11 | Runs AssetVault (Flask, port 8080) as a dedicated low-privilege account, `svc_vault` |
| Client | 10.0.1.12 | Legitimate user, interacts via browser |
| Kali (attacker) | 10.0.1.20 | Attacker machine, second NIC on NAT for tooling only |

**Intended workflow:** AssetVault is a small internal file-storage tool.
Users upload files of any type through a browser UI and can later
"open" a stored file to preview its contents. That's the entire intended
feature set - deliberately simple, so the vulnerability and its fix are
easy to isolate and explain.

## 2. The Vulnerability & Why It's Relevant

**Primary weakness - CWE-434, Unrestricted Upload of File with Dangerous
Type (leading to Remote Code Execution):** The `/open` route decided how
to handle an uploaded file based on its extension, executing `.py` and
`.sh` files via `subprocess` instead of just reading them. Combined with
an intentionally unrestricted upload endpoint, this let an attacker upload
a reverse-shell script and trigger code execution as `svc_vault`.

**Secondary weakness - sudo misconfiguration:** `svc_vault` had a sudoers
entry granting `NOPASSWD: /usr/bin/python3` with no restriction, letting
any process running as that account escalate straight to root
(`sudo python3 -c 'import os; os.system("/bin/bash")'`).

**Why this pairing was chosen:** Unrestricted file upload is one of the
most common real-world initial-access vectors, and pairing it with an
unrestricted-interpreter sudo rule mirrors a very common (and very
dangerous) real misconfiguration pattern - the same class of issue
catalogued by projects like GTFOBins. Both are single-point-of-failure
bugs: easy to spot once you know what to look for, and each independently
critical, which makes for a clean before/after demonstration and a
genuine two-layer defense-in-depth patch.

## 3. Logging & Journaling Implementation

Three layers, deliberately covering different visibility gaps:

1. **journald** - Cince AssetVault runs as a systemd service
   (`assetvault.service`), service lifecycle events, stdout/stderr, and
   restarts are queryable via `journalctl -u assetvault`.
2. **Application log (`/var/log/assetvault.log`)** - Python's `logging`
   module records every upload and open request, including the raw
   filename parameter. This is the layer that captures the *entry point*
   of an attack - but nothing beyond it, since it can only log from
   inside a live Flask request.
3. **auditd** - Kernel-level, keyed watch rules covering what the app log
   structurally cannot see:
   ```
   -w /etc/sudoers.d/ -p wa -k sudoers_change
   -w /opt/assetvault/reports -p wa -k assetvault_files
   -w /usr/bin/python3 -p x -k python_exec
   -a always,exit -F arch=b64 -S execve -F path=/bin/bash -k bash_spawn
   ```
   This is what actually captures privilege escalation: once a reverse
   shell connects, everything that happens inside it (`sudo -l`, the
   `sudo python3` escalation) is invisible to Flask/`assetvault.log` but
   fully visible to `ausearch -k python_exec` / `-k bash_spawn`.

**Key finding from this layering:** Application logs and OS-level audit
logs cover different parts of an attack chain. Relying on app logs alone
would have left the privilege-escalation stage completely undocumented -
a genuine investigative gap that only a second, independent logging layer
closes.

## 4. Evidence of the Attack

1. Malicious file (`shell.py`, a Python reverse-shell payload) uploaded
   via the unrestricted `/upload` endpoint - accepted with no validation.
2. `/open` triggered on `shell.py` - `assetvault.log` records:
   ```
   INFO open request from 10.0.1.20 filename='shell.py'
   ERROR open error: Command ['python3', '/opt/assetvault/reports/shell.py'] timed out after 5 seconds
   ```
   The timeout is expected - the spawned reverse shell blocks, connects
   out to a `nc -lvnp 4444` listener on the client, and the Flask request
   ends without further visibility.
3. Shell caught on the listener, running as `svc_vault` (`id` confirms
   `uid=997(svc_vault)`).
4. `sudo -l` reveals the misconfigured rule; `sudo python3 -c 'import os; os.system("/bin/bash")'` escalates to `uid=0(root)`.
5. `sudo ausearch -k python_exec` and `-k bash_spawn` show the escalation
   at the OS level, filling the gap the app log couldn't.

## 5. Patch / Mitigation Strategy

**Fix 1 - remove the execution path entirely (root cause, not a
blocklist):** rather than filtering which file types are allowed to
upload, `/open` was rewritten to never invoke an interpreter under any
circumstance. It only ever opens a file in binary mode and returns a
text preview of the first 2000 bytes. `subprocess` is no longer imported
at all - there is no code path left capable of executing anything,
regardless of file extension or content. Uploads remain intentionally
unrestricted, since the app's storage requirement (any file type) is
still valid; the fix guarantees stored data is never treated as code.
A path-traversal guard (`os.path.realpath` containment check) was added
as a bonus hardening measure found during the fix.

**Fix 2 - remove the sudo misconfiguration (independent root cause):**
```bash
sudo rm /etc/sudoers.d/vault
```
This is deliberately treated as a separate fix, not made redundant by
Fix 1: even though Fix 1 alone breaks this specific attack chain, the
sudo rule is a standalone privilege-escalation path that would remain
exploitable by *any* other means of reaching the `svc_vault` account.
Patching only the entry point and leaving this in place would be
single-point remediation, not real hardening.

## 6. Proof the Mitigation Works

- **Upload still succeeds** for any file type (confirms legitimate
  functionality is unaffected) - `shell.py` uploads without error.
- **Execution attempt fails safely:** clicking "Open" on `shell.py` now
  returns a plain-text preview of the Python source code. No shell
  spawns; the `nc -lvnp 4444` listener never receives a connection.
- **`assetvault.log` shows no timeout/error** for the request - it
  completes normally, because the file is only ever read, never run.
- **Privilege escalation is independently blocked:** even connecting as
  `svc_vault` through another means (e.g. SSH) and running `sudo -l`
  returns `svc_vault is not allowed to run sudo on server` - the rule
  is gone, closing the second door regardless of the first.
- **Normal usage unaffected:** legitimate `.txt`/`.csv` uploads and
  opens behave identically before and after the patch.

## 7. Architecture

![AssetVault lab architecture](architecture.svg)

Three VMs on an isolated internal network. The client and the Kali
attacker machine both reach the server's AssetVault service on port
8080 - one with legitimate upload/open requests, the other with a
malicious upload and reverse shell. The server runs three logging
layers (journald, the application log, and auditd), each covering a
different part of the attack chain. The sudoers misconfiguration is
called out separately, since it's an independent flaw patched on its
own merits rather than as a byproduct of fixing the upload
vulnerability.

## 8. Repository Contents

- `app.py` - patched AssetVault source (see git history / `app_vulnerable.py`
  for the pre-patch version used in the attack demo)
- `assetvault.service` - systemd unit
- `audit_rules/assetvault.rules` - auditd watch rules
- `hash_chain_logger.py` - optional tamper-evident wrapper for
  `assetvault.log`
- `architecture.svg` - architecture diagram (embedded above)
- `docs/` - this README, presentation notes

## 9. Lessons / Design Notes

- Storage and execution should always be architecturally separate;
  extension-based execution decisions are a recurring, avoidable bug
  class.
- A single logging layer is not sufficient to reconstruct a full attack
  chain - application and OS-level logging answer different questions
  and both are needed for investigation.
- Fixing the vulnerability that was actually exploited does not mean
  the system is secure; independent misconfigurations found along the
  way (here, the sudoers rule) should be fixed on their own merits.
