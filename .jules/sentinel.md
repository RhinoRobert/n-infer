## 2024-05-18 - [MIME Sniffing and Content-Type enforcement]
**Vulnerability:** Found lack of MIME sniffing protection and relaxed content-type checking in the local chat example server.
**Learning:** `readJson` accepted arbitrary `Content-Type`, which could allow CSRF or unexpected request body handling if integrated elsewhere. Missing `x-content-type-options` allowed potential browser MIME sniffing.
**Prevention:** Add `x-content-type-options: nosniff` header to API endpoints. Enforce `application/json` Content-Type for endpoints expecting JSON request bodies.
