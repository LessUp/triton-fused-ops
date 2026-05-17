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
PERFORMANCE_CHART_PATH = THEME_ROOT / "components" / "PerformanceChart.vue"
SYSTEM_BLUEPRINT_PATH = THEME_ROOT / "components" / "SystemBlueprint.vue"
BENCHMARK_VISUALIZATION_PATHS = {
    "en": DOCS_ROOT / "en" / "guides" / "benchmark-visualization.md",
    "zh": DOCS_ROOT / "zh" / "guides" / "benchmark-visualization.md",
}
BENCHMARK_FIGURES_PATH = THEME_ROOT / "components" / "BenchmarkVisualizationFigures.vue"
HOMEPAGE_PATHS = {
    "en": DOCS_ROOT / "en" / "index.md",
    "zh": DOCS_ROOT / "zh" / "index.md",
}
WHITEPAPER_HOMEPAGE_COMPONENTS = (
    "WhitepaperHero",
    "ReaderTracks",
    "KernelAtlas",
    "SystemBlueprint",
    "ResearchLandscape",
)
ENGLISH_WHITEPAPER_COMPONENTS = (
    *WHITEPAPER_HOMEPAGE_COMPONENTS,
)
ENGLISH_WHITEPAPER_PAGES = {
    DOCS_ROOT / "en" / "overview" / "index.md": (
        "Kernel family",
        "Benchmarking",
        "Auto-Tuning",
        "Performance metrics",
    ),
    DOCS_ROOT / "en" / "academy" / "index.md": (
        "Academy map",
        "/en/academy/system-overview",
        "/en/kernel-families/",
        "/en/architecture-lab/",
    ),
    DOCS_ROOT / "en" / "academy" / "system-overview.md": (
        "Public API surface",
        "Validation contracts",
        "Kernel and reference execution",
        "Benchmarking and Auto-Tuning",
    ),
    DOCS_ROOT / "en" / "kernel-families" / "index.md": (
        "Fused RMSNorm + RoPE",
        "Fused Gated MLP",
        "FP8 GEMM",
        "FP8 quantization utilities",
    ),
    DOCS_ROOT / "en" / "kernel-families" / "rmsnorm-rope.md": (
        "RMSNorm",
        "RoPE",
        "validation",
        "Benchmarking",
    ),
    DOCS_ROOT / "en" / "kernel-families" / "gated-mlp.md": (
        "SwiGLU",
        "GeGLU",
        "intermediate_dim",
        "activation",
    ),
    DOCS_ROOT / "en" / "kernel-families" / "fp8-stack.md": (
        "FP8 GEMM",
        "quantize_fp8",
        "dequantize_fp8",
        "scale",
    ),
    DOCS_ROOT / "en" / "architecture-lab" / "index.md": (
        "module map",
        "runtime contracts",
        "public exports",
        "validation",
    ),
    DOCS_ROOT / "en" / "architecture-lab" / "module-map.md": (
        "triton_ops/__init__.py",
        "triton_ops/kernels/",
        "triton_ops/reference/",
        "triton_ops/benchmark/",
    ),
    DOCS_ROOT / "en" / "architecture-lab" / "runtime-contracts.md": (
        "DeviceError",
        "ShapeMismatchError",
        "UnsupportedDtypeError",
        "contiguous",
    ),
    DOCS_ROOT / "en" / "reference-research" / "index.md": (
        "related projects",
        "references",
        "evolution thinking",
        "research agenda",
    ),
    DOCS_ROOT / "en" / "reference-research" / "related-projects.md": (
        "OpenAI Triton",
        "PyTorch",
        "vLLM",
        "TensorRT-LLM",
    ),
    DOCS_ROOT / "en" / "reference-research" / "references.md": (
        "FlashAttention",
        "FP8 Formats for Deep Learning",
        "RoFormer",
        "Root Mean Square Layer Normalization",
    ),
    DOCS_ROOT / "en" / "reference-research" / "evolution-thinking.md": (
        "industrial",
        "kernel family",
        "evidence-backed",
        "next questions",
    ),
}
ENGLISH_GUIDE_EXPECTATIONS = {
    DOCS_ROOT / "en" / "guides" / "index.md": (
        "Choose the narrative you need",
        "./performance",
        "./integration",
        "../reference-research/",
    ),
    DOCS_ROOT / "en" / "guides" / "performance.md": (
        "Benchmarking",
        "Auto-Tuning",
        "Performance metrics",
        "BenchmarkSuite",
    ),
    DOCS_ROOT / "en" / "guides" / "integration.md": (
        "runtime contracts",
        "Kernel family",
        "FusedRMSNormRoPE",
        "FP8Linear",
    ),
}
CHINESE_WHITEPAPER_PAGES = {
    DOCS_ROOT / "zh" / "overview" / "index.md": (
        "Kernel family",
        "Benchmarking",
        "Auto-Tuning",
        "Performance metrics",
    ),
    DOCS_ROOT / "zh" / "academy" / "index.md": (
        "学院地图",
        "/zh/academy/system-overview",
        "/zh/kernel-families/",
        "/zh/architecture-lab/",
    ),
    DOCS_ROOT / "zh" / "academy" / "system-overview.md": (
        "公共 API",
        "验证契约",
        "Kernel 与 reference",
        "Benchmarking",
    ),
    DOCS_ROOT / "zh" / "kernel-families" / "index.md": (
        "Fused RMSNorm + RoPE",
        "Fused Gated MLP",
        "FP8 GEMM",
        "FP8 量化工具",
    ),
    DOCS_ROOT / "zh" / "kernel-families" / "rmsnorm-rope.md": (
        "RMSNorm",
        "RoPE",
        "验证",
        "Benchmarking",
    ),
    DOCS_ROOT / "zh" / "kernel-families" / "gated-mlp.md": (
        "SwiGLU",
        "GeGLU",
        "intermediate_dim",
        "activation",
    ),
    DOCS_ROOT / "zh" / "kernel-families" / "fp8-stack.md": (
        "FP8 GEMM",
        "quantize_fp8",
        "dequantize_fp8",
        "scale",
    ),
    DOCS_ROOT / "zh" / "architecture-lab" / "index.md": (
        "模块地图",
        "运行时契约",
        "公共导出",
        "validation",
    ),
    DOCS_ROOT / "zh" / "architecture-lab" / "module-map.md": (
        "triton_ops/__init__.py",
        "triton_ops/kernels/",
        "triton_ops/reference/",
        "triton_ops/benchmark/",
    ),
    DOCS_ROOT / "zh" / "architecture-lab" / "runtime-contracts.md": (
        "DeviceError",
        "ShapeMismatchError",
        "UnsupportedDtypeError",
        "contiguous",
    ),
    DOCS_ROOT / "zh" / "reference-research" / "index.md": (
        "相关项目",
        "参考文献",
        "演进思路",
        "研究议程",
    ),
    DOCS_ROOT / "zh" / "reference-research" / "related-projects.md": (
        "OpenAI Triton",
        "PyTorch",
        "vLLM",
        "TensorRT-LLM",
    ),
    DOCS_ROOT / "zh" / "reference-research" / "references.md": (
        "FlashAttention",
        "FP8 Formats for Deep Learning",
        "RoFormer",
        "Root Mean Square Layer Normalization",
    ),
    DOCS_ROOT / "zh" / "reference-research" / "evolution-thinking.md": (
        "工业化",
        "kernel family",
        "evidence-backed",
        "下一个问题",
    ),
}
CHINESE_GUIDE_EXPECTATIONS = {
    DOCS_ROOT / "zh" / "guides" / "index.md": (
        "选择你需要的叙事路径",
        "./performance",
        "./integration",
        "../reference-research/",
    ),
    DOCS_ROOT / "zh" / "guides" / "performance.md": (
        "Benchmarking",
        "Auto-Tuning",
        "Performance metrics",
        "BenchmarkSuite",
    ),
    DOCS_ROOT / "zh" / "guides" / "integration.md": (
        "运行时契约",
        "Kernel family",
        "FusedRMSNormRoPE",
        "FP8Linear",
    ),
}
ENGLISH_BASE_AWARE_CARD_LINKS = {
    DOCS_ROOT / "en" / "index.md": (
        'href="./overview/"',
        'href="./academy/"',
        'href="./guides/"',
    ),
    DOCS_ROOT / "en" / "overview" / "index.md": (
        'href="../academy/"',
        'href="../kernel-families/"',
        'href="../architecture-lab/"',
    ),
    DOCS_ROOT / "en" / "guides" / "index.md": (
        'href="./integration"',
        'href="./performance"',
        'href="../reference-research/"',
    ),
    DOCS_ROOT / "en" / "reference-research" / "index.md": (
        'href="./related-projects"',
        'href="./references"',
        'href="./evolution-thinking"',
    ),
}
CHINESE_BASE_AWARE_CARD_LINKS = {
    DOCS_ROOT / "zh" / "index.md": (
        'href="./overview/"',
        'href="./academy/"',
        'href="./guides/"',
    ),
    DOCS_ROOT / "zh" / "overview" / "index.md": (
        'href="../academy/"',
        'href="../kernel-families/"',
        'href="../architecture-lab/"',
    ),
    DOCS_ROOT / "zh" / "guides" / "index.md": (
        'href="./integration"',
        'href="./performance"',
        'href="../reference-research/"',
    ),
    DOCS_ROOT / "zh" / "reference-research" / "index.md": (
        'href="./related-projects"',
        'href="./references"',
        'href="./evolution-thinking"',
    ),
}


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

        for component in WHITEPAPER_HOMEPAGE_COMPONENTS:
            assert f"import {component} from '@theme/components/{component}.vue'" in contents
            assert f"<{component}" in contents


def test_english_homepage_uses_whitepaper_landing_composition():
    contents = read_text(HOMEPAGE_PATHS["en"])

    assert "HomeHero" not in contents

    for component in ENGLISH_WHITEPAPER_COMPONENTS:
        assert f"import {component} from '@theme/components/{component}.vue'" in contents
        assert f"<{component}" in contents


def test_chinese_homepage_uses_localized_whitepaper_landing_composition():
    contents = read_text(HOMEPAGE_PATHS["zh"])

    assert "HomeHero" not in contents

    for component in WHITEPAPER_HOMEPAGE_COMPONENTS:
        assert f"import {component} from '@theme/components/{component}.vue'" in contents
        assert f"<{component}" in contents


def test_english_whitepaper_routes_are_not_stubs_and_cover_key_topics():
    forbidden_stub_markers = (
        "reserved for",
        "will be expanded",
        "route hub is in place",
        "follow-up content tasks",
    )

    for path, expected_snippets in ENGLISH_WHITEPAPER_PAGES.items():
        contents = read_text(path)
        lowered = contents.lower()

        for marker in forbidden_stub_markers:
            assert marker not in lowered, f"{path} should not remain a placeholder"

        for snippet in expected_snippets:
            assert snippet in contents, f"{path} should mention {snippet!r}"


def test_chinese_whitepaper_routes_are_not_stubs_and_cover_key_topics():
    forbidden_stub_markers = (
        "预留给",
        "后续内容任务",
        "先为新信息架构",
        "后续会补齐",
    )

    for path, expected_snippets in CHINESE_WHITEPAPER_PAGES.items():
        contents = read_text(path)

        for marker in forbidden_stub_markers:
            assert marker not in contents, f"{path} should not remain a placeholder"

        for snippet in expected_snippets:
            assert snippet in contents, f"{path} should mention {snippet!r}"


def test_english_guides_match_whitepaper_information_architecture():
    for path, expected_snippets in ENGLISH_GUIDE_EXPECTATIONS.items():
        contents = read_text(path)

        for snippet in expected_snippets:
            assert snippet in contents, f"{path} should mention {snippet!r}"


def test_chinese_guides_match_whitepaper_information_architecture():
    for path, expected_snippets in CHINESE_GUIDE_EXPECTATIONS.items():
        contents = read_text(path)

        for snippet in expected_snippets:
            assert snippet in contents, f"{path} should mention {snippet!r}"


def test_english_whitepaper_card_links_are_base_aware():
    for path, expected_links in ENGLISH_BASE_AWARE_CARD_LINKS.items():
        contents = read_text(path)

        assert 'href="/en/' not in contents, f"{path} should not hardcode /en card links"

        for link in expected_links:
            assert link in contents, f"{path} should include base-aware link {link!r}"


def test_chinese_whitepaper_card_links_are_base_aware():
    for path, expected_links in CHINESE_BASE_AWARE_CARD_LINKS.items():
        contents = read_text(path)

        assert 'href="/zh/' not in contents, f"{path} should not hardcode /zh card links"

        for link in expected_links:
            assert link in contents, f"{path} should include base-aware link {link!r}"


def test_english_guides_sidebar_lists_child_pages():
    contents = read_text(DOCS_ROOT / ".vitepress" / "config.ts")

    assert "const enSidebar = {" in contents
    assert "text: 'Guides'" in contents
    assert "link: '/en/guides/'" in contents
    assert "link: '/en/guides/integration'" in contents
    assert "link: '/en/guides/performance'" in contents
    assert "link: '/en/guides/benchmark-visualization'" in contents


def test_chinese_sidebar_lists_new_whitepaper_children():
    contents = read_text(DOCS_ROOT / ".vitepress" / "config.ts")

    assert "text: '学院'" in contents
    assert "link: '/zh/academy/system-overview'" in contents
    assert "link: '/zh/kernel-families/rmsnorm-rope'" in contents
    assert "link: '/zh/kernel-families/gated-mlp'" in contents
    assert "link: '/zh/kernel-families/fp8-stack'" in contents
    assert "link: '/zh/architecture-lab/module-map'" in contents
    assert "link: '/zh/architecture-lab/runtime-contracts'" in contents
    assert "link: '/zh/reference-research/related-projects'" in contents
    assert "link: '/zh/reference-research/references'" in contents
    assert "link: '/zh/reference-research/evolution-thinking'" in contents
    assert "link: '/zh/guides/integration'" in contents
    assert "link: '/zh/guides/performance'" in contents
    assert "link: '/zh/guides/benchmark-visualization'" in contents


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


def test_performance_chart_reads_theme_safe_palette_tokens():
    contents = read_text(PERFORMANCE_CHART_PATH)

    expected_tokens = [
        "--editorial-accent",
        "--editorial-signal",
        "--editorial-rule",
        "--vp-c-text-1",
        "--vp-c-bg-elv",
    ]

    assert "getComputedStyle" in contents

    for token in expected_tokens:
        assert token in contents


def test_benchmark_figures_pass_microsecond_units_to_theme_safe_chart():
    chart_contents = read_text(PERFORMANCE_CHART_PATH)
    figure_contents = read_text(BENCHMARK_FIGURES_PATH)

    assert "valueUnit" in chart_contents
    assert "yAxisLabel" in chart_contents
    assert 'value-unit="µs"' in figure_contents


def test_system_blueprint_uses_shared_figure_frame_language():
    contents = read_text(SYSTEM_BLUEPRINT_PATH)

    assert "FigureFrame" in contents
    assert "tone=\"accent\"" in contents


def test_benchmark_visualization_pages_use_figure_frame_and_theme_safe_tokens():
    forbidden_literals = (
        "#76B900",
        "#30363d",
        "#21262d",
        "#3476f6",
        "#1a4a9e",
        "#ffc517",
        "#c49000",
        "#ff5454",
        "#0d1117",
        "#fff",
        "#1a1a1a",
        "rgba(118,185,0,0.08)",
    )

    figure_contents = read_text(BENCHMARK_FIGURES_PATH)

    assert "FigureFrame" in figure_contents
    assert "--viz-accent" in figure_contents

    for literal in forbidden_literals:
        assert literal not in figure_contents, f"benchmark figures should not hard-code {literal}"

    for locale, path in BENCHMARK_VISUALIZATION_PATHS.items():
        contents = read_text(path)
        assert "BenchmarkVisualizationFigures" in contents, (
            f"{locale} benchmark visualization page should render the shared figure component"
        )
