import mongoose from 'mongoose';

const fileSchema = new mongoose.Schema({
    originalFileName: {
        type: String,
        required: true
    },
    storedFileName: {
        type: String,
        required: true,
        maxlength: 100,
        default: null
    },
    fileSize: {
        type: Number,
        required: true
    },
    MIMEType: {
        type: String,
        required: true
    },
    uploadDate: {   
        type: Date,
        default: Date.now,
        required: true
    }

});

const fileFormat = mongoose.model("Format", fileSchema);

export default fileFormat;