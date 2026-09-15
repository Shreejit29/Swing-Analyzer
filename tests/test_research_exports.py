"""
Tests for public exports from src.research.

These tests ensure that newly integrated research components remain
available through the package-level API.
"""

from __future__ import annotations

import src.research as research


def test_market_context_exports():
    assert hasattr(
        research,
        "MarketContextConfig",
    )

    assert hasattr(
        research,
        "MarketContextResult",
    )

    assert hasattr(
        research,
        "build_market_context",
    )

    assert hasattr(
        research,
        "build_benchmark_features",
    )

    assert hasattr(
        research,
        "add_relative_strength",
    )


def test_market_data_context_exports():
    assert hasattr(
        research,
        "MarketContextRequest",
    )

    assert hasattr(
        research,
        "MarketContextData",
    )

    assert hasattr(
        research,
        "MarketContextDataLoader",
    )

    assert hasattr(
        research,
        "load_market_context",
    )

    assert hasattr(
        research,
        "build_stock_market_context",
    )


def test_market_integration_exports():
    assert hasattr(
        research,
        "MarketIntegrationConfig",
    )

    assert hasattr(
        research,
        "MarketIntegrationResult",
    )

    assert hasattr(
        research,
        "integrate_market_context",
    )

    assert hasattr(
        research,
        "add_market_context_to_stock",
    )

    assert hasattr(
        research,
        "market_integration_summary",
    )


def test_unified_feature_exports():
    assert hasattr(
        research,
        "UnifiedFeatureConfig",
    )

    assert hasattr(
        research,
        "UnifiedFeatureResult",
    )

    assert hasattr(
        research,
        "build_unified_features",
    )

    assert hasattr(
        research,
        "feature_matrix",
    )

    assert hasattr(
        research,
        "unified_feature_names",
    )

    assert hasattr(
        research,
        "unified_feature_summary",
    )


def test_market_context_classes_are_importable():
    config = research.MarketContextConfig()

    assert isinstance(
        config,
        research.MarketContextConfig,
    )


def test_market_integration_config_is_importable():
    config = research.MarketIntegrationConfig()

    assert isinstance(
        config,
        research.MarketIntegrationConfig,
    )


def test_unified_feature_config_is_importable():
    config = research.UnifiedFeatureConfig()

    assert isinstance(
        config,
        research.UnifiedFeatureConfig,
    )


def test_existing_research_exports_remain_available():
    expected = [
        "ResearchPipeline",
        "ResearchPipelineResult",
        "ResearchStage",
        "ResearchDatasetBuilder",
        "LeakageAuditor",
        "TemporalSplitter",
        "WalkForwardResearchEngine",
        "ModelDevelopmentOrchestrator",
        "ResearchExperimentRunner",
        "ControlledModelSelector",
        "SelectionPipeline",
        "SelectionRegistry",
        "ModelCard",
        "ModelCardRegistry",
        "CalibrationResearchEngine",
        "RangeResearchEngine",
        "RegimeResearchEngine",
        "BacktestResearchEngine",
        "RobustnessResearchEngine",
        "ResearchEvidence",
        "ResearchEvidenceCollector",
        "FinalHoldoutGate",
        "FinalHoldoutStage",
        "ProtectedHoldoutPipeline",
        "ResearchPipelineIntegrator",
        "ProductionPredictionGateway",
        "MultiHorizonPredictionEngine",
    ]

    for name in expected:
        assert hasattr(
            research,
            name
        ), (
            f"Missing public research export: "
            f"{name}"
        )


def test_research_package_does_not_expose_production_approval_as_feature_generation():
    """
    Feature generation must remain separate from production approval.

    This is an architectural boundary test rather than a claim that
    approval functionality does not exist elsewhere in the package.
    """

    assert hasattr(
        research,
        "build_unified_features",
    )

    assert hasattr(
        research,
        "ResearchApprovalBridge",
    )

    assert (
        research.build_unified_features
        is not research.ResearchApprovalBridge
    )


def test_public_api_is_deterministic():
    first = sorted(
        name
        for name in dir(research)
        if not name.startswith("_")
    )

    second = sorted(
        name
        for name in dir(research)
        if not name.startswith("_")
    )

    assert first == second
