# Open Problems: starter-hugo-academic

This document catalogs open problems, infrastructure misconfigurations, and content audit items for the **starter-hugo-academic** portfolio repository (`config/`, `content/`, `layouts/`, `.github/`).

---

## 1. Algorithmic & Implementation Problems

- **Cloudflare Pages vs. Netlify Build Configuration (`HIGH #2`)**
  - **Problem**: `netlify.toml` is present and pins Hugo to `0.111.3` (from 2023), but production deployment is hosted on Cloudflare Pages (`ichor.pages.dev`).
  - **Proposed Approach**: Reconcile build pipelines between Cloudflare Pages and Netlify, updating to a supported Hugo release without breaking existing Wowchemy v5 asset bundles.
- **Go Module & Wowchemy Theme Stagnation (`HIGH #3`)**
  - **Problem**: `go.mod` specifies `go 1.15`, and the Wowchemy module (`v5.7.1-0.20230420205746`) is over three years old.
  - **Proposed Approach**: Test upgrading `go.mod` to `go 1.21+` and evaluate whether to maintain stable Wowchemy v5 or plan a migration to HugoBlox v2.

---

## 2. Bugs & Unresolved Issues

- **Base URL Configuration Mismatch (`HIGH #1`)**
  - **Problem**: `config/_default/config.yaml` sets `baseURL: "https://jeffreymorais.netlify.app/"`, whereas the site is deployed to `https://ichor.pages.dev/`. This generates incorrect canonical URLs, sitemap hostnames, and OpenGraph metadata.
- **Copywriting Non-Compliance in Project Briefs (`FAIL #5, #6`)**
  - **Problem**: `content/project/leonne/index.md` and `content/project/quantumcrypto/index.md` use banned corporate terminology (e.g., "leveraging") and vague phrasing ("enhancing security and efficiency").
  - **Context**: Tracked in `audit-2026-04-18-full-review.md`. Requires copywriting cleanup to use specific, metric-grounded technical descriptions.
- **Missing Project Repository Links (`FAIL #8`)**
  - **Problem**: `content/project/basiq/index.md` lacks GitHub repository or live demo links.

---

## 3. Theoretical & Scientific Problems

- **Technical Precision in Cryptographic and Physics Abstracts**
  - **Problem**: Ensuring project summaries for theoretical cryptography (`content/project/qring`) and quantum photonics (`content/project/casimir`) accurately reflect implemented prototypes versus theoretical proposals without overclaiming.

---

## 4. Code Maintenance & Refactoring Opportunities

- **Dead GitHub Actions Workflow (`MEDIUM #4`)**
  - **Opportunity**: `.github/workflows/updater-wip.yml` includes `if: github.repository_owner == 'wowchemy'`, rendering it inactive on `IsolatedSingularity` forks. Should be removed or updated.
- **Sponsorship Link Attribution (`MEDIUM #5`)**
  - **Opportunity**: `.github/FUNDING.yml` directs sponsors to `gcushen` and Wowchemy rather than the repository owner.
- **`.gitignore` Deduplication (`LOW #8`)**
  - **Opportunity**: Remove duplicate entries for `.env`, `audit-*.md`, and `Jenova/` in `.gitignore`.
