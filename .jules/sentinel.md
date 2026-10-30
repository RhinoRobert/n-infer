## 2024-10-31 - Missing nosniff headers on API endpoints
**Vulnerability:** The API endpoints (`/api/info`, `/api/load`, `/api/chat`) were missing the `x-content-type-options: nosniff` header, which was only applied to static file serving.
**Learning:** It's easy to forget security headers on dynamic API responses (like JSON or event streams) when manually setting `writeHead`, focusing only on static file routes.
**Prevention:** Always apply basic security headers like `x-content-type-options: nosniff` in a centralized response handler or middleware rather than just on specific GET file routes.
