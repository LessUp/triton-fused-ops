#!/usr/bin/env node
/**
 * Sync CHANGELOG.md to docs/en/release-notes/changelog.md and zh/release-notes/changelog.md
 *
 * This script copies the content from the root CHANGELOG.md to the docs site,
 * with formatting changes for better documentation presentation.
 *
 * Run from the docs directory: node scripts/sync-changelog.mjs
 */

import { readFileSync, writeFileSync, existsSync, mkdirSync } from "fs";
import { dirname, join } from "path";
import { fileURLToPath } from "url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const docsDir = join(__dirname, "..");
const rootDir = join(docsDir, "..");

const sourcePath = join(rootDir, "CHANGELOG.md");

// Check if source exists
if (!existsSync(sourcePath)) {
  console.log("No CHANGELOG.md found at root, skipping sync");
  process.exit(0);
}

const HEADER_EN = `# Changelog

This page documents the changes in each release of Triton Fused Ops.

`;

const HEADER_ZH = `# 变更日志

本页面记录 Triton Fused Ops 每个版本的变更。

`;

// Read the source file
let content = readFileSync(sourcePath, "utf-8");

// Remove the HTML comment block at the top
content = content.replace(/<!--[\s\S]*?-->\n*/g, "");

// Remove the "# Changelog" title (we'll add our own header)
content = content.replace(/^# Changelog\n+/, "");
content = content.replace(/^# 变更日志\n+/, "");

// Convert title format: ## [0.1.0] - 2024-01-15 -> ## 0.1.0 (2024-01-15)
content = content.replace(
  /^## \[([^\]]+)\] - (\d{4}-\d{1,2}-\d{1,2})/gm,
  "## $1 ($2)"
);

// Remove subsection headers like ### Added, ### Changed, ### Fixed
content = content.replace(/^### (Added|Changed|Fixed|Improved|Features|Bug Fixes|Breaking)\n+/gm, "");

// Ensure release-notes directories exist
const enDir = join(docsDir, "en", "release-notes");
const zhDir = join(docsDir, "zh", "release-notes");

if (!existsSync(enDir)) mkdirSync(enDir, { recursive: true });
if (!existsSync(zhDir)) mkdirSync(zhDir, { recursive: true });

// Write the target files
const enTargetPath = join(enDir, "changelog.md");
const zhTargetPath = join(zhDir, "changelog.md");

writeFileSync(enTargetPath, HEADER_EN + content.trim() + "\n");
writeFileSync(zhTargetPath, HEADER_ZH + content.trim() + "\n");

console.log(`Synced changelog to ${enTargetPath}`);
console.log(`Synced changelog to ${zhTargetPath}`);
