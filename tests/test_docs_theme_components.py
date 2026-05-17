from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = REPO_ROOT / "docs"
THEME_ROOT = REPO_ROOT / "docs" / ".vitepress" / "theme"
STYLE_PATH = THEME_ROOT / "style.css"
INDEX_PATH = THEME_ROOT / "index.ts"
COMPONENTS = {
    "HomeHero.vue": {"needs_locale": True},
    "WhitepaperHero.vue": {"needs_locale": True},
    "ReaderTracks.vue": {"needs_locale": True},
    "KernelAtlas.vue": {"needs_locale": True},
    "SystemBlueprint.vue": {"needs_locale": True},
    "FigureFrame.vue": {"needs_locale": False},
    "ResearchLandscape.vue": {"needs_locale": True},
}
HOMEPAGE_PATHS = {
    "en": DOCS_ROOT / "en" / "index.md",
    "zh": DOCS_ROOT / "zh" / "index.md",
}
HOMEPAGE_EDITORIAL_COMPONENTS = (
    "HomeHero",
    "ReaderTracks",
    "KernelAtlas",
    "SystemBlueprint",
)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_editorial_theme_tokens_and_component_styles_exist():
    css = read_text(STYLE_PATH)

    expected_tokens = [
        "--editorial-ink",
        "--editorial-paper",
        "--editorial-accent",
        "--editorial-signal",
        "--editorial-shadow",
    ]
    expected_blocks = [
        ".whitepaper-hero",
        ".reader-tracks",
        ".kernel-atlas",
        ".system-blueprint",
        ".figure-frame",
        ".research-landscape",
    ]

    for token in expected_tokens:
        assert token in css

    for selector in expected_blocks:
        assert selector in css


def test_shared_editorial_components_are_production_ready():
    for filename, expectations in COMPONENTS.items():
        contents = read_text(THEME_ROOT / "components" / filename)
        assert "docs-shell-stub" not in contents, f"{filename} should not remain a stub"
        assert "<script setup lang=\"ts\">" in contents, f"{filename} should use typed setup"

        if expectations["needs_locale"]:
            assert "useData" in contents, f"{filename} should read VitePress locale data"

    for filename in ("HomeHero.vue", "WhitepaperHero.vue", "ReaderTracks.vue", "KernelAtlas.vue"):
        contents = read_text(THEME_ROOT / "components" / filename)
        assert "withBase" in contents, f"{filename} should generate base-aware internal links"


def test_home_hero_uses_editorial_hero_layout_without_metrics_strip():
    contents = read_text(THEME_ROOT / "components" / "HomeHero.vue")

    assert "metrics-strip" not in contents
    assert "whitepaper-hero" in contents
    assert "toLocalizedHref" in contents
    assert '"/en/' not in contents
    assert '"/zh/' not in contents


def test_homepages_compose_editorial_shared_components_without_legacy_blocks():
    for locale, path in HOMEPAGE_PATHS.items():
        contents = read_text(path)

        assert "KernelShowcase" not in contents, f"{locale} homepage should not use legacy showcase"
        assert "ArchitecturePreview" not in contents, (
            f"{locale} homepage should not use legacy architecture preview"
        )

        for component in HOMEPAGE_EDITORIAL_COMPONENTS:
            assert f"import {component} from '@theme/components/{component}.vue'" in contents
            assert f"<{component} />" in contents


def test_kernel_showcase_localized_links_are_base_aware():
    contents = read_text(THEME_ROOT / "components" / "KernelShowcase.vue")

    assert "withBase" in contents
    assert "toLocalizedHref" in contents
    assert '"/en/' not in contents
    assert '"/zh/' not in contents


def test_theme_registers_shared_editorial_components():
    index_source = read_text(INDEX_PATH)

    for component_name in COMPONENTS:
        stem = component_name.removesuffix(".vue")
        assert f"import {stem} from './components/{component_name}'" in index_source
        assert f"app.component('{stem}', {stem})" in index_source
