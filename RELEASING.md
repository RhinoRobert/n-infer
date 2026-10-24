# Releasing

How a version gets from `main` to npm. The pipeline is `.github/workflows/release.yml`; this page is the human side of it.

## The first release, in order

Publishing is permanent (npm versions can never be re-used), so the first one goes step by step:

1. **Open a pull request** from `release-prep` into `main` and wait for CI to pass on Linux, macOS and Windows. Fix whatever it finds on the branch. Merge when it is green.
2. **npm organization:** create `@n-infer` (see the setup below). The unscoped name `n-infer` and the scope were free when checked.
3. **GitHub settings and the one-time token** (see the setup below).
4. **Dry run:** Actions -> Release -> _Run workflow_ with dry-run on. (The button only appears once `release.yml` is on `main`.) It builds all binaries and checks every package, publishing nothing. Download the artifacts if you want to inspect them.
5. **Tag:** `git tag v0.1.0-beta.0 && git push origin v0.1.0-beta.0`, then approve the `npm-publish` environment.
6. **Verify** from a clean folder (commands below), then check `npm view n-infer dist-tags`. If `latest` is missing (a first release made only under the `beta` tag), run `npm dist-tag add n-infer@0.1.0-beta.0 latest` so a plain `npm install n-infer` works too.
7. **Switch to trusted publishing** and delete the token (setup step 2).

The Linux CUDA add-on is marked experimental: if its build fails, the release still goes out without it (the README says so). Check the build log and package size before promoting it.

## One-time setup (maintainer)

1. **npm account** with 2FA (done), then create the free **`@n-infer` organization** at npmjs.com so no one else can claim those package names.
2. **First publish.** npm trusted publishing is configured per package, so each package must exist before it can be linked to this repository. Publish the first version of each package once:
   - add a short-lived _granular access token_ (read/write, `n-infer` and `@n-infer/*`, 2FA bypass for automation) as the repository secret `NPM_TOKEN`, and run the release;
   - then, for every package (`n-infer` and each `@n-infer/<target>`), open its **Settings -> Trusted Publisher** on npmjs.com and link `RhinoRobert/n-infer`, workflow `release.yml`, environment `npm-publish`;
   - delete the `NPM_TOKEN` secret. Later releases use OIDC only.
3. **GitHub settings:** create an environment named `npm-publish` with yourself as a required reviewer; protect `main` (pull request + passing CI); turn on secret scanning, Dependabot alerts and private vulnerability reporting.

## Cutting a release

1. Update `CHANGELOG.md`.
2. Bump the version in **both** `package.json` and `Cargo.toml` (and the `optionalDependencies` / `peerDependencies` versions in `package.json` to match). The pipeline refuses a release where they disagree.
3. Rehearse: run the **Release** workflow by hand with _dry-run_ on. It builds every binary, stages every package, checks their contents, and runs `npm publish --dry-run`. Nothing is published.
4. Tag and push: `git tag v0.1.0-beta.1 && git push origin v0.1.0-beta.1`.
5. Approve the `npm-publish` environment when prompted. A version containing `-` (like `-beta.1`) is published under the `beta` dist-tag, so `npm install n-infer` keeps resolving to the last stable release.
6. Verify from a clean machine:

```bash
npm view n-infer dist-tags
mkdir try && cd try && npm init -y && npm install n-infer@beta
node -e "import('n-infer').then(m => console.log(m.backend, m.hardware().physicalCores))"
npm audit signatures
```

## Promoting a beta

When a beta has been exercised, publish the same code as a stable version (drop the suffix, tag, repeat), or point `latest` at it with `npm dist-tag add n-infer@<version> latest`.

## If a publish fails part-way

npm versions are immutable and a multi-package publish is not atomic. If some platform packages published and the main package did not, **do not reuse the version**: fix the cause, bump the patch (or beta counter) and release again. Deprecate any half-published version with `npm deprecate`.
