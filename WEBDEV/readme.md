# Web Development Head

# Build a File Upload & Storage Service

## Overview

Build a backend service that allows users to upload, store, retrieve,
and manage files. The focus of this project is on backend engineering,
system design, and writing clean, maintainable code.

You may use any programming language, framework, or database.

------------------------------------------------------------------------

## Functional Requirements

Your application must support the following:

-   Upload files
-   Download files
-   Delete files
-   View/List uploaded files
-   Store file metadata in a database

The metadata should include at least:

-   File ID
-   Original Filename
-   Stored Filename
-   File Size
-   MIME Type
-   Upload Timestamp

------------------------------------------------------------------------

## Design Requirements

Your implementation should demonstrate thoughtful engineering decisions.

Some questions to consider while designing your solution:

-   How are uploaded files stored?
-   How do you prevent filename collisions?
-   How do you organize uploaded files?
-   How does your application validate uploaded files?
-   What happens if the database operation succeeds but the file write
    fails?
-   How would you scale this service in the future?

There is **no single correct implementation**. We are interested in your
reasoning, tradeoffs, and design decisions.

------------------------------------------------------------------------

## Constraints

### ❌ Not Allowed

The following are **not allowed**, as they abstract away the core
functionality being evaluated:

-   UploadThing
-   Appwrite Storage
-   Supabase Storage
-   Firebase Storage
-   Cloudinary
-   Any library or service that completely abstracts file storage

### ✅ Allowed

You may use libraries/frameworks for:

-   HTTP routing
-   Multipart parsing (e.g. Multer, Busboy)
-   Database ORM/query builders
-   Authentication
-   Validation
-   Image processing
-   Logging

You are expected to implement the storage layer yourself.

------------------------------------------------------------------------

## Bonus Features (Optional)

You may implement any of the following:

-   Authentication & Authorization
-   File Search
-   Pagination
-   Folder Support
-   File Versioning
-   Image Previews / Thumbnails
-   Background Processing
-   Signed URLs
-   Rate Limiting
-   Unit Tests
-   Docker Support
-   Cloud Storage Abstraction (S3, GCS, etc.)
-   Caching

------------------------------------------------------------------------

## Deliverables

Submit the following:

-   GitHub Repository
-   README containing:
    -   Project Overview
    -   Setup Instructions
    -   Architecture Overview
    -   Design Decisions
    -   Assumptions Made
    -   Future Improvements
-   Architecture / System Design Diagram
-   A junior-level workshop-style presentation on the project
-   A technical video walkthrough explaining your approach, implementation, and key design decisions
-   A README.md technical document included in your GitHub submission

------------------------------------------------------------------------

## Submission Guidelines

-   Push your project to a public GitHub repository.
-   Ensure your repository contains complete setup instructions.
-   Your project should run without requiring modifications.
-   Clearly document any assumptions or limitations.

------------------------------------------------------------------------

## Important Notes

-   We are **not** evaluating the number of features you implement.
-   We value thoughtful engineering decisions over feature count.
-   Simplicity is preferred over unnecessary complexity.
-   Your code should be clean, maintainable, and well-structured.

Good luck, and happy building!
