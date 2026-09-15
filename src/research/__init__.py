"""
AI Swing Analyser — Research Package.

Public API for the research, validation, model-selection,
market-context, feature-integration, and production-boundary layers.

Important:
This package exposes research functionality but does not itself
grant production approval.
"""

from .config import (
    ResearchValidationConfig,
    ResearchFeatureConfig,
    ResearchHyperparameterConfig,
    ResearchCalibrationConfig,
    ResearchRangeConfig,
    ResearchBacktestConfig,
    ResearchRobustnessConfig,
    ResearchApprovalConfig,
    ResearchPipelineConfig,
)

from .research_config import (
    IntegratedValidationConfig,
    IntegratedCalibrationConfig,
    IntegratedRangeConfig,
    IntegratedRegimeConfig,
    IntegratedBacktestConfig,
    IntegratedRobustnessConfig,
    IntegratedHoldoutConfig,
    IntegratedApprovalConfig,
    IntegratedTradingConfig,
    IntegratedResearchConfig,
    default_integrated_research_config,
)

from .pipeline import (
    ResearchPipeline,
    ResearchPipelineResult,
    ResearchStage,
)

from .dataset_builder import (
    ResearchDatasetBuilder,
    ResearchDatasetResult,
    build_research_dataset,
)

from .leakage_audit import (
    LeakageFinding,
    LeakageAuditReport,
    LeakageAuditConfig,
    LeakageAuditor,
    audit_research_dataset,
    assert_leakage_free,
    compare_future_mutation,
)

from .temporal_split import (
    TemporalSplit,
    HoldoutSplit,
    TemporalSplitter,
    create_holdout_split,
    create_walk_forward_splits,
)

from .model_development import (
    DevelopmentPartitions,
    ModelDevelopmentResult,
    ModelDevelopmentOrchestrator,
)

from .experiment_runner import (
    ExperimentCandidate,
    ExperimentRunResult,
    ResearchExperimentRunner,
)

from .walk_forward_research import (
    WalkForwardFoldResult,
    WalkForwardResearchResult,
    WalkForwardResearchEngine,
    run_walk_forward_research,
)

from .holdout_evaluation import (
    HoldoutEvaluationResult,
    FinalHoldoutEvaluator,
    evaluate_final_holdout,
)

from .calibration_research import (
    CalibrationResearchResult,
    CalibrationResearchEngine,
    calibrate_research_probabilities,
)

from .range_research import (
    RangeResearchResult,
    RangeResearchEngine,
    run_range_research,
)

from .regime_research import (
    RegimeResearchResult,
    RegimeResearchEngine,
    run_regime_research,
)

from .backtest_research import (
    BacktestResearchResult,
    BacktestResearchEngine,
    calculate_return_distribution,
    stress_trade_returns,
)

from .robustness_research import (
    MonteCarloResult,
    StressResult,
    TradeRemovalResult,
    RobustnessResearchResult,
    RobustnessResearchEngine,
    run_robustness_research,
)

from .model_selection import (
    ModelCandidateScore,
    ModelSelectionResult,
    ModelSelectionConfig,
    ControlledModelSelector,
    select_best_model,
)

from .selection_pipeline import (
    SelectionCandidate,
    SelectionPipelineResult,
    SelectionPipeline,
    run_model_selection,
)

from .selection_registry import (
    SelectionRecord,
    SelectionRegistry,
    register_selection,
)

from .selection_stage import (
    ModelSelectionStageResult,
    ModelSelectionStage,
    run_model_selection_stage,
)

from .model_card import (
    ModelCard,
    ModelCardBuilder,
    model_card_from_selection,
    save_model_card,
    load_model_card,
)

from .model_card_registry import (
    ModelCardRegistry,
    register_model_card,
)

from .holdout_gate import (
    HoldoutGateStatus,
    HoldoutGateInput,
    HoldoutGateResult,
    FinalHoldoutGate,
    create_holdout_gate,
    evaluate_holdout_gate,
)

from .holdout_stage import (
    HoldoutStageResult,
    FinalHoldoutStage,
    run_final_holdout_stage,
)

from .pipeline_holdout import (
    HoldoutPipelineResult,
    ProtectedHoldoutPipeline,
    run_protected_holdout,
)

from .final_holdout_evidence import (
    FINAL_HOLDOUT_EVIDENCE_NAME,
    adapt_final_holdout_result,
    attach_final_holdout_evidence,
    final_holdout_evidence_summary,
)

from .evidence_collector import (
    WALK_FORWARD_STAGE,
    FINAL_HOLDOUT_STAGE,
    CALIBRATION_STAGE,
    RANGE_STAGE,
    REGIME_STAGE,
    BACKTEST_STAGE,
    ROBUSTNESS_STAGE,
    LEAKAGE_STAGE,
    REQUIRED_EVIDENCE_STAGES,
    EvidenceCollectionResult,
    ResearchEvidenceCollector,
    collect_research_evidence,
    collect_final_holdout_evidence,
    attach_final_holdout_to_evidence,
    build_final_holdout_evidence,
    final_holdout_passed,
    final_holdout_evaluated,
    final_holdout_status,
    final_holdout_summary,
    validate_final_holdout_identity,
    collect_holdout_evidence_map,
)

from .integration import (
    EvidenceStatus,
    EvidenceItem,
    ResearchEvidence,
    ResearchEvidenceBuilder,
    create_research_evidence,
    assert_research_ready,
)

from .stage_adapters import (
    adapt_walk_forward_result,
    adapt_holdout_result,
    adapt_calibration_result,
    adapt_range_result,
    adapt_regime_result,
    adapt_backtest_result,
    adapt_robustness_result,
    adapt_leakage_result,
    adapt_stage_result,
)

from .approval_bridge import (
    ApprovalBridgeResult,
    ResearchApprovalBridge,
    evaluate_research_approval,
    assert_research_approved,
)

from .pipeline_integration import (
    IntegrationStatus,
    IntegrationStage,
    IntegratedResearchResult,
    ResearchPipelineIntegrator,
    create_research_integrator,
    integrate_final_holdout,
)

from .production_prediction import (
    ProductionPredictionStatus,
    ProductionPredictionConfig,
    ProductionPrediction,
    ProductionPredictionGateway,
)

from .multi_horizon_prediction import (
    DEFAULT_HORIZONS as DEFAULT_PRODUCTION_HORIZONS,
    MultiHorizonConfig,
    MultiHorizonPrediction,
    MultiHorizonPredictionEngine,
)

from .market_context import (
    MarketContextConfig,
    MarketContextResult,
    build_benchmark_features,
    build_market_context,
    add_relative_strength,
    market_strength_score,
)

from .market_data_context import (
    MarketContextRequest,
    MarketContextData,
    MarketContextDataLoader,
    load_market_context,
    build_stock_market_context,
)

from .market_integration import (
    MarketIntegrationConfig,
    MarketIntegrationResult,
    integrate_market_context,
    add_market_context_to_stock,
    market_integration_summary,
)

from .feature_integration import (
    UnifiedFeatureConfig,
    UnifiedFeatureResult,
    build_unified_features,
    feature_matrix,
    unified_feature_names,
    unified_feature_summary,
)


__all__ = [
    # Legacy research configuration
    "ResearchValidationConfig",
    "ResearchFeatureConfig",
    "ResearchHyperparameterConfig",
    "ResearchCalibrationConfig",
    "ResearchRangeConfig",
    "ResearchBacktestConfig",
    "ResearchRobustnessConfig",
    "ResearchApprovalConfig",
    "ResearchPipelineConfig",

    # Integrated configuration
    "IntegratedValidationConfig",
    "IntegratedCalibrationConfig",
    "IntegratedRangeConfig",
    "IntegratedRegimeConfig",
    "IntegratedBacktestConfig",
    "IntegratedRobustnessConfig",
    "IntegratedHoldoutConfig",
    "IntegratedApprovalConfig",
    "IntegratedTradingConfig",
    "IntegratedResearchConfig",
    "default_integrated_research_config",

    # Core pipeline
    "ResearchPipeline",
    "ResearchPipelineResult",
    "ResearchStage",

    # Dataset and leakage
    "ResearchDatasetBuilder",
    "ResearchDatasetResult",
    "build_research_dataset",
    "LeakageFinding",
    "LeakageAuditReport",
    "LeakageAuditConfig",
    "LeakageAuditor",
    "audit_research_dataset",
    "assert_leakage_free",
    "compare_future_mutation",

    # Temporal research
    "TemporalSplit",
    "HoldoutSplit",
    "TemporalSplitter",
    "create_holdout_split",
    "create_walk_forward_splits",

    # Model development
    "DevelopmentPartitions",
    "ModelDevelopmentResult",
    "ModelDevelopmentOrchestrator",
    "ExperimentCandidate",
    "ExperimentRunResult",
    "ResearchExperimentRunner",

    # Walk-forward and evaluation
    "WalkForwardFoldResult",
    "WalkForwardResearchResult",
    "WalkForwardResearchEngine",
    "run_walk_forward_research",
    "HoldoutEvaluationResult",
    "FinalHoldoutEvaluator",
    "evaluate_final_holdout",

    # Research stages
    "CalibrationResearchResult",
    "CalibrationResearchEngine",
    "calibrate_research_probabilities",
    "RangeResearchResult",
    "RangeResearchEngine",
    "run_range_research",
    "RegimeResearchResult",
    "RegimeResearchEngine",
    "run_regime_research",
    "BacktestResearchResult",
    "BacktestResearchEngine",
    "calculate_return_distribution",
    "stress_trade_returns",
    "MonteCarloResult",
    "StressResult",
    "TradeRemovalResult",
    "RobustnessResearchResult",
    "RobustnessResearchEngine",
    "run_robustness_research",

    # Model selection
    "ModelCandidateScore",
    "ModelSelectionResult",
    "ModelSelectionConfig",
    "ControlledModelSelector",
    "select_best_model",
    "SelectionCandidate",
    "SelectionPipelineResult",
    "SelectionPipeline",
    "run_model_selection",
    "SelectionRecord",
    "SelectionRegistry",
    "register_selection",
    "ModelSelectionStageResult",
    "ModelSelectionStage",
    "run_model_selection_stage",

    # Model cards
    "ModelCard",
    "ModelCardBuilder",
    "model_card_from_selection",
    "save_model_card",
    "load_model_card",
    "ModelCardRegistry",
    "register_model_card",

    # Final holdout protection
    "HoldoutGateStatus",
    "HoldoutGateInput",
    "HoldoutGateResult",
    "FinalHoldoutGate",
    "create_holdout_gate",
    "evaluate_holdout_gate",
    "HoldoutStageResult",
    "FinalHoldoutStage",
    "run_final_holdout_stage",
    "HoldoutPipelineResult",
    "ProtectedHoldoutPipeline",
    "run_protected_holdout",

    # Final holdout evidence
    "FINAL_HOLDOUT_EVIDENCE_NAME",
    "adapt_final_holdout_result",
    "attach_final_holdout_evidence",
    "final_holdout_evidence_summary",

    # Evidence collection
    "WALK_FORWARD_STAGE",
    "FINAL_HOLDOUT_STAGE",
    "CALIBRATION_STAGE",
    "RANGE_STAGE",
    "REGIME_STAGE",
    "BACKTEST_STAGE",
    "ROBUSTNESS_STAGE",
    "LEAKAGE_STAGE",
    "REQUIRED_EVIDENCE_STAGES",
    "EvidenceCollectionResult",
    "ResearchEvidenceCollector",
    "collect_research_evidence",
    "collect_final_holdout_evidence",
    "attach_final_holdout_to_evidence",
    "build_final_holdout_evidence",
    "final_holdout_passed",
    "final_holdout_evaluated",
    "final_holdout_status",
    "final_holdout_summary",
    "validate_final_holdout_identity",
    "collect_holdout_evidence_map",

    # Evidence framework
    "EvidenceStatus",
    "EvidenceItem",
    "ResearchEvidence",
    "ResearchEvidenceBuilder",
    "create_research_evidence",
    "assert_research_ready",

    # Stage adapters
    "adapt_walk_forward_result",
    "adapt_holdout_result",
    "adapt_calibration_result",
    "adapt_range_result",
    "adapt_regime_result",
    "adapt_backtest_result",
    "adapt_robustness_result",
    "adapt_leakage_result",
    "adapt_stage_result",

    # Approval boundary
    "ApprovalBridgeResult",
    "ResearchApprovalBridge",
    "evaluate_research_approval",
    "assert_research_approved",

    # Integrated pipeline
    "IntegrationStatus",
    "IntegrationStage",
    "IntegratedResearchResult",
    "ResearchPipelineIntegrator",
    "create_research_integrator",
    "integrate_final_holdout",

    # Production prediction
    "ProductionPredictionStatus",
    "ProductionPredictionConfig",
    "ProductionPrediction",
    "ProductionPredictionGateway",
    "DEFAULT_PRODUCTION_HORIZONS",
    "MultiHorizonConfig",
    "MultiHorizonPrediction",
    "MultiHorizonPredictionEngine",

    # Indian market context
    "MarketContextConfig",
    "MarketContextResult",
    "build_benchmark_features",
    "build_market_context",
    "add_relative_strength",
    "market_strength_score",
    "MarketContextRequest",
    "MarketContextData",
    "MarketContextDataLoader",
    "load_market_context",
    "build_stock_market_context",

    # Market integration
    "MarketIntegrationConfig",
    "MarketIntegrationResult",
    "integrate_market_context",
    "add_market_context_to_stock",
    "market_integration_summary",

    # Unified features
    "UnifiedFeatureConfig",
    "UnifiedFeatureResult",
    "build_unified_features",
    "feature_matrix",
    "unified_feature_names",
    "unified_feature_summary",
]
