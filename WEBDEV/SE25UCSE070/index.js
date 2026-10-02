import express from "express";
import { fileURLToPath } from "url";
import path from "path";
import mongoose from "mongoose";
import fileFormat from "./db/file.js";
import multer from "multer";
import fs from "fs";
import methodOverride from "method-override";

const app = express();
const port = 8080;

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const uploadDir = path.join(__dirname, "uploads");
fs.mkdirSync(uploadDir, { recursive: true });

const upload = multer({
    dest: uploadDir,
    limits: { fileSize: 20 * 1024 * 1024 },
});

app.set("views", path.join(__dirname, "views"));
app.set("view engine", "ejs");

app.use(express.static(path.join(__dirname, "public")));
app.use(methodOverride('_method')); 

main().then(() => {
    console.log("connection successful")
})
.catch(err => console.log(err));

async function main(){
    await mongoose.connect('mongodb://127.0.0.1:27017/files');
}

app.listen(port, "127.0.0.1", () => {
    console.log(`listening on port ${port}`);
});

app.get("/", async (req, res) => {
    try {
        const fileFormats = await fileFormat.find().sort({ uploadDate: -1 });
        res.render("home", { fileFormats });
    } catch (err) {
        console.log(err);
        res.status(500).send("Could not load files");
    }
});

app.get("/upload", (req, res) => {
    res.render("new");
});

app.post("/upload", upload.array("file", 10), async (req, res) => {
    if (!req.files || req.files.length === 0) {
        return res.status(400).send("No files uploaded");
    }

    const savedFiles = [];

    for (const file of req.files) {
        const ext = path.extname(file.originalname).toLowerCase();

        const _id = new mongoose.Types.ObjectId();
        const now = new Date();
        const year = String(now.getFullYear());
        const month = String(now.getMonth() + 1).padStart(2, "0");
        const subDir = path.join(uploadDir, year, month);
        fs.mkdirSync(subDir, { recursive: true });

        const storedFileName = path.join(year, month, `${_id}${ext}`);
        const finalPath = path.join(uploadDir, storedFileName);

        fs.renameSync(file.path, finalPath);

        const newFile = new fileFormat({
            _id,
            originalFileName: file.originalname,
            storedFileName,
            fileSize: file.size,
            MIMEType: file.mimetype,
        });

        try {
            await newFile.save();
            savedFiles.push(file.originalname);
        } catch (err) {
            console.log(err);
            fs.unlinkSync(finalPath);
        }
    }

    res.redirect("/");
});

app.get("/files/:id/download", async (req, res) => {
    let { id } = req.params;
    const file = await fileFormat.findById(id);
    if (!file) return res.status(404).send("File not found");

    const filePath = path.join(uploadDir, file.storedFileName);
    res.download(filePath, file.originalFileName);
});

app.get("/files/:id/view", async (req, res) => {
    let { id } = req.params;
    const file = await fileFormat.findById(id);
    if (!file) return res.status(404).send("File not found");

    const filePath = path.join(uploadDir, file.storedFileName);
    res.sendFile(filePath);
});

app.delete("/files/:id", async (req, res) => {
    let { id } = req.params;
    const file = await fileFormat.findByIdAndDelete(id);
    if (!file) return res.status(404).send("File not found");

    const filePath = path.join(uploadDir, file.storedFileName);
    fs.unlink(filePath, (err) => {
        if (err) console.log("Failed to delete file from disk:", err);
    });

    res.redirect("/");
});