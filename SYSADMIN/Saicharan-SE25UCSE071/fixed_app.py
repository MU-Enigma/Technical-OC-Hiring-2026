# /opt/assetvault/app.py
# AssetVault -- internal report storage/viewer (PATCHED VERSION)
#
# Fix summary:
#   - CWE-434 (Unrestricted Upload of File with Dangerous Type / execution):
#     the /open route no longer invokes any interpreter (python3, bash, etc.)
#     under any circumstance. Uploaded content is always treated as inert
#     data and only ever read + previewed as text, never executed.
#   - Upload itself remains intentionally unrestricted (any file type is
#     still accepted) since the fix is about guaranteeing nothing uploaded
#     can ever run, not about filtering what can be stored.
#   - Path traversal guard added on /open so a filename like
#     "../../etc/passwd" can't be used to read files outside REPORT_DIR.
from flask import Flask, request, render_template_string
import os, logging

app = Flask(__name__)
REPORT_DIR = "/opt/assetvault/reports"
os.makedirs(REPORT_DIR, exist_ok=True)

logging.basicConfig(
    filename="/var/log/assetvault.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

PAGE = """
<!doctype html>
<html>
<head>
    <title>AssetVault</title>
    <style>
        body { font-family: sans-serif; max-width: 700px; margin: 40px auto; }
        fieldset { margin-bottom: 25px; padding: 15px; }
        input[type=text] { width: 300px; padding: 5px; }
        input[type=submit] { padding: 6px 16px; }
        pre { background: #f4f4f4; padding: 10px; white-space: pre-wrap; word-break: break-all; }
        h1 { margin-bottom: 5px; }
        .sub { color: #666; margin-top: 0; }
    </style>
</head>
<body>
    <h1>AssetVault</h1>
    <p class="sub">Internal report storage & viewer</p>

    <fieldset>
        <legend>Upload a report (any file type)</legend>
        <form action="/upload" method="post" enctype="multipart/form-data">
            <input type="file" name="file">
            <input type="submit" value="Upload">
        </form>
    </fieldset>

    <fieldset>
        <legend>Open a report (read-only preview)</legend>
        <form action="/open" method="post">
            <label>Filename (as uploaded):</label><br>
            <input type="text" name="filename" placeholder="report1.txt">
            <input type="submit" value="Open">
        </form>
    </fieldset>

    {% if result %}
    <fieldset>
        <legend>Result</legend>
        <pre>{{ result }}</pre>
    </fieldset>
    {% endif %}

    <fieldset>
        <legend>Files in vault</legend>
        <pre>{{ files }}</pre>
    </fieldset>
</body>
</html>
"""

def list_files():
    try:
        return "\n".join(sorted(os.listdir(REPORT_DIR))) or "(empty)"
    except Exception as e:
        return f"error listing files: {e}"

@app.route("/")
def index():
    return render_template_string(PAGE, result=None, files=list_files())

@app.route("/upload", methods=["POST"])
def upload():
    """
    Intentionally unrestricted: any file type is accepted, since this is
    meant to function as general-purpose report storage. Safety comes from
    /open never executing content -- not from filtering uploads.
    """
    f = request.files.get("file")
    if not f or f.filename == "":
        return render_template_string(PAGE, result="No file selected.", files=list_files())

    # basename() strips any directory components -- prevents writing
    # outside REPORT_DIR via a crafted filename like "../../etc/cron.d/x"
    safe_name = os.path.basename(f.filename)
    save_path = os.path.join(REPORT_DIR, safe_name)
    f.save(save_path)
    logging.info(f"upload from {request.remote_addr} saved={save_path}")

    return render_template_string(PAGE, result=f"Saved to {save_path}", files=list_files())

@app.route("/open", methods=["POST"])
def open_report():
    """
    PATCHED: no interpreter is ever invoked, regardless of file extension
    or content. 'Opening' a report only reads and previews raw bytes as
    text. Uploaded data is always data, never code.
    """
    filename = request.form.get("filename", "")
    logging.info(f"open request from {request.remote_addr} filename={filename!r}")

    # basename() again strips any path components from user input
    safe_name = os.path.basename(filename)
    path = os.path.join(REPORT_DIR, safe_name)

    # Belt-and-braces path traversal check: resolved path must still be
    # inside REPORT_DIR after resolution
    real_report_dir = os.path.realpath(REPORT_DIR)
    real_path = os.path.realpath(path)
    if not real_path.startswith(real_report_dir + os.sep):
        logging.warning(f"REJECTED open (path escape attempt) from {request.remote_addr}: {filename!r}")
        return render_template_string(PAGE, result="Invalid filename.", files=list_files())

    if not os.path.isfile(path):
        return render_template_string(PAGE, result="File not found.", files=list_files())

    try:
        with open(path, "rb") as fh:
            raw = fh.read(2000)
        preview = raw.decode("utf-8", errors="replace")
        output = f"(read-only preview, first 2000 bytes)\n\n{preview}"
    except Exception as e:
        logging.error(f"open error: {e}")
        output = f"Error: {e}"

    return render_template_string(PAGE, result=output, files=list_files())

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
