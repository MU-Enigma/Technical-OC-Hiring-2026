# /opt/assetvault/app.py
# AssetVault -- internal report storage/viewer (deliberately vulnerable for lab purposes)
from flask import Flask, request, render_template_string
import subprocess, os, logging

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
        pre { background: #f4f4f4; padding: 10px; white-space: pre-wrap; }
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
        <legend>Open a report</legend>
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
    meant to function as general-purpose report storage.
    """
    f = request.files.get("file")
    if not f or f.filename == "":
        return render_template_string(PAGE, result="No file selected.", files=list_files())

    safe_name = os.path.basename(f.filename)
    save_path = os.path.join(REPORT_DIR, safe_name)
    f.save(save_path)
    logging.info(f"upload from {request.remote_addr} saved={save_path}")

    return render_template_string(PAGE, result=f"Saved to {save_path}", files=list_files())

@app.route("/open", methods=["POST"])
def open_report():
    """
    VULNERABILITY (CWE-434): 'Opening' a report executes it via an
    interpreter chosen by file extension, with no check on what the
    file actually contains. Any uploaded file can become code execution.
    """
    filename = request.form.get("filename", "")
    logging.info(f"open request from {request.remote_addr} filename={filename!r}")

    path = os.path.join(REPORT_DIR, filename)
    if not os.path.isfile(path):
        return render_template_string(PAGE, result="File not found.", files=list_files())

    try:
        if filename.endswith(".py"):
            result = subprocess.run(["python3", path], capture_output=True, text=True, timeout=5)
        elif filename.endswith(".sh"):
            result = subprocess.run(["bash", path], capture_output=True, text=True, timeout=5)
        else:
            with open(path, errors="replace") as fh:
                return render_template_string(PAGE, result=fh.read(2000), files=list_files())
        output = f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    except Exception as e:
        logging.error(f"open error: {e}")
        output = f"Error: {e}"

    return render_template_string(PAGE, result=output, files=list_files())

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
