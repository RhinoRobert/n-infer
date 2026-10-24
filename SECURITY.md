# Security policy

n-infer runs a large C++ inference engine inside your Node.js process, so security reports are taken seriously.

## Reporting a vulnerability

Please **do not open a public issue**. Use GitHub's private reporting:
[Security -> Report a vulnerability](https://github.com/RhinoRobert/n-infer/security/advisories/new).

Include what you found, how to reproduce it, and the versions involved (`n-infer`, Node.js, OS, `backend`). You will get an acknowledgement within a few days, and we will agree a disclosure timeline together.

## Supported versions

Only the latest release (and the latest beta while no stable release exists) receives fixes.

## What is in scope

- Memory-safety or crash bugs reachable from the JavaScript API with valid or hostile **options and prompts**
- Ways for message or prompt text to become control tokens when `parseSpecial` is off
- Bypasses of the input limits, the request queue, or `close()` / abort semantics
- Problems in how the native binary is located, loaded or version-matched
- Release pipeline and supply-chain weaknesses (workflows, published file contents)

## What is not

- A model that follows instructions written in plain language inside a prompt (language-level prompt injection). No tokenizer can prevent this: treat model output as untrusted.
- Vulnerabilities in a malicious **model file**. GGUF files are parsed by llama.cpp's C++ code, so treat model files like executable code and only load ones you trust. `modelSha256` lets you pin the exact file.
- Vulnerabilities in llama.cpp itself: report those upstream (we will update the pinned version once a fix exists).

See [docs/THREAT_MODEL.md](docs/THREAT_MODEL.md) for the full analysis.
