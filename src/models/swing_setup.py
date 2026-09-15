"""
Swing-trading setup construction for AI Swing Analyser.

This module converts validated model outputs into a structured swing setup.

The module does NOT decide whether the underlying model is trustworthy.
That responsibility belongs to the validation and gating layers.

It calculates:
    - Entry
    - Stop loss
    - Target prices
    - Predicted price range
    - Risk/reward
    - Position sizing
    - Capital allocation
    - Expected return
    - Setup quality

All calculations are deterministic and can later be connected to the
decision/gating engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional

import math


class SetupDirection(str, Enum):
    """Direction of a swing setup."""

    LONG = "LONG"
    SHORT = "SHORT"
    NONE = "NONE"


class SetupStatus(str, Enum):
    """Status of a constructed swing setup."""

    VALID = "VALID"
    INVALID = "INVALID"
    NO_TRADE = "NO_TRADE"


@dataclass
class SwingSetupConfig:
    """
    Configuration for swing setup construction.

    Defaults are conservative research starting points. They must be
    validated through historical backtesting before being used live.
    """

    max_risk_per_trade: float = 0.01

    default_stop_loss_pct: float = 0.03

    minimum_stop_loss_pct: float = 0.01
    maximum_stop_loss_pct: float = 0.08

    minimum_risk_reward: float = 2.0

    target_1_fraction: float = 0.50
    target_2_fraction: float = 1.00

    maximum_capital_fraction: float = 0.25

    allow_long: bool = True
    allow_short: bool = False

    # Position sizing is based on risk rather than arbitrary share count.
    use_risk_based_position_sizing: bool = True

    def __post_init__(self) -> None:
        if not 0.0 < self.max_risk_per_trade <= 1.0:
            raise ValueError(
                "max_risk_per_trade must be between 0 and 1."
            )

        if not 0.0 < self.default_stop_loss_pct < 1.0:
            raise ValueError(
                "default_stop_loss_pct must be between 0 and 1."
            )

        if not 0.0 < self.minimum_stop_loss_pct < 1.0:
            raise ValueError(
                "minimum_stop_loss_pct must be between 0 and 1."
            )

        if not 0.0 < self.maximum_stop_loss_pct < 1.0:
            raise ValueError(
                "maximum_stop_loss_pct must be between 0 and 1."
            )

        if (
            self.minimum_stop_loss_pct
            > self.maximum_stop_loss_pct
        ):
            raise ValueError(
                "minimum_stop_loss_pct cannot exceed "
                "maximum_stop_loss_pct."
            )

        if self.minimum_risk_reward <= 0:
            raise ValueError(
                "minimum_risk_reward must be positive."
            )

        if not 0.0 < self.target_1_fraction <= 1.0:
            raise ValueError(
                "target_1_fraction must be between 0 and 1."
            )

        if self.target_2_fraction < self.target_1_fraction:
            raise ValueError(
                "target_2_fraction must be >= target_1_fraction."
            )

        if not 0.0 < self.maximum_capital_fraction <= 1.0:
            raise ValueError(
                "maximum_capital_fraction must be between 0 and 1."
            )


@dataclass
class SwingSetupInput:
    """
    Model and market information required to construct a swing setup.
    """

    direction: SetupDirection

    current_price: float

    predicted_return: Optional[float] = None

    lower_return: Optional[float] = None
    median_return: Optional[float] = None
    upper_return: Optional[float] = None

    confidence: float = 0.0

    support_price: Optional[float] = None
    resistance_price: Optional[float] = None

    atr: Optional[float] = None

    risk_reward: Optional[float] = None

    trade_allowed: bool = False

    metadata: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.current_price <= 0:
            raise ValueError(
                "current_price must be positive."
            )

        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                "confidence must be between 0 and 1."
            )

        if self.atr is not None and self.atr < 0:
            raise ValueError(
                "ATR cannot be negative."
            )


@dataclass
class SwingSetup:
    """
    Fully constructed swing-trading setup.
    """

    status: SetupStatus

    direction: SetupDirection

    entry_price: Optional[float]

    stop_loss: Optional[float]

    target_1: Optional[float]
    target_2: Optional[float]

    predicted_lower_price: Optional[float]
    predicted_median_price: Optional[float]
    predicted_upper_price: Optional[float]

    risk_per_share: Optional[float]

    reward_to_target_1: Optional[float]
    reward_to_target_2: Optional[float]

    risk_reward_target_1: Optional[float]
    risk_reward_target_2: Optional[float]

    position_size: Optional[int]

    capital_required: Optional[float]

    maximum_risk_amount: Optional[float]

    expected_return: Optional[float]

    confidence: float

    quality_score: float

    reason: str

    warnings: list[str] = field(default_factory=list)

    metadata: Dict[str, object] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        """Whether the setup passed construction checks."""
        return self.status == SetupStatus.VALID

    @property
    def is_tradeable(self) -> bool:
        """Whether the setup is currently considered tradable."""
        return self.status == SetupStatus.VALID


class SwingSetupEngine:
    """
    Constructs swing setups from validated model outputs.

    The engine is deliberately conservative:
        1. Invalid market/model inputs are rejected.
        2. Risk/reward must meet the configured threshold.
        3. Position size is risk-based.
        4. Capital allocation is capped.
        5. The model's trade_allowed flag is respected.
    """

    def __init__(
        self,
        config: Optional[SwingSetupConfig] = None,
    ) -> None:
        self.config = config or SwingSetupConfig()

    def build(
        self,
        inputs: SwingSetupInput,
        account_size: Optional[float] = None,
    ) -> SwingSetup:
        """
        Construct a swing setup.

        Parameters
        ----------
        inputs:
            Validated prediction and market information.

        account_size:
            Available trading capital. Required for position sizing.

        Returns
        -------
        SwingSetup
            Structured setup.
        """

        warnings: list[str] = []

        # --------------------------------------------------------------
        # Basic validation
        # --------------------------------------------------------------

        if inputs.direction == SetupDirection.NONE:
            return self._invalid_setup(
                inputs,
                reason="No directional prediction is available.",
            )

        if (
            inputs.direction == SetupDirection.LONG
            and not self.config.allow_long
        ):
            return self._no_trade_setup(
                inputs,
                reason="Long trading is disabled.",
            )

        if (
            inputs.direction == SetupDirection.SHORT
            and not self.config.allow_short
        ):
            return self._no_trade_setup(
                inputs,
                reason="Short trading is disabled.",
            )

        if not inputs.trade_allowed:
            return self._no_trade_setup(
                inputs,
                reason=(
                    "The upstream research decision engine has not "
                    "authorized this setup for trading."
                ),
            )

        if account_size is not None and account_size <= 0:
            return self._invalid_setup(
                inputs,
                reason="account_size must be positive.",
            )

        # --------------------------------------------------------------
        # Entry
        # --------------------------------------------------------------

        entry = float(inputs.current_price)

        # --------------------------------------------------------------
        # Predicted price range
        # --------------------------------------------------------------

        lower_price = self._return_to_price(
            entry,
            inputs.lower_return,
        )

        median_price = self._return_to_price(
            entry,
            (
                inputs.median_return
                if inputs.median_return is not None
                else inputs.predicted_return
            ),
        )

        upper_price = self._return_to_price(
            entry,
            inputs.upper_return,
        )

        # --------------------------------------------------------------
        # Stop loss
        # --------------------------------------------------------------

        stop_loss = self._calculate_stop_loss(
            inputs=inputs,
            entry=entry,
        )

        if stop_loss is None:
            return self._invalid_setup(
                inputs,
                reason="Unable to construct a valid stop-loss.",
            )

        risk_per_share = abs(entry - stop_loss)

        if risk_per_share <= 0:
            return self._invalid_setup(
                inputs,
                reason="Risk per share is zero.",
            )

        # --------------------------------------------------------------
        # Target construction
        # --------------------------------------------------------------

        target_1, target_2 = self._calculate_targets(
            inputs=inputs,
            entry=entry,
            stop_loss=stop_loss,
            predicted_median_price=median_price,
            predicted_upper_price=upper_price,
            predicted_lower_price=lower_price,
        )

        # --------------------------------------------------------------
        # Risk/reward
        # --------------------------------------------------------------

        reward_1 = self._reward(
            direction=inputs.direction,
            entry=entry,
            target=target_1,
        )

        reward_2 = self._reward(
            direction=inputs.direction,
            entry=entry,
            target=target_2,
        )

        rr_1 = self._safe_divide(
            reward_1,
            risk_per_share,
        )

        rr_2 = self._safe_divide(
            reward_2,
            risk_per_share,
        )

        # Target 2 is the primary decision target.
        primary_rr = rr_2

        if primary_rr is None:
            return self._invalid_setup(
                inputs,
                reason="Unable to calculate risk/reward.",
            )

        if primary_rr < self.config.minimum_risk_reward:
            return self._no_trade_setup(
                inputs,
                reason=(
                    f"Risk/reward {primary_rr:.2f} is below the "
                    f"minimum required "
                    f"{self.config.minimum_risk_reward:.2f}."
                ),
                entry=entry,
                stop_loss=stop_loss,
                target_1=target_1,
                target_2=target_2,
                predicted_lower_price=lower_price,
                predicted_median_price=median_price,
                predicted_upper_price=upper_price,
                risk_per_share=risk_per_share,
                reward_to_target_1=reward_1,
                reward_to_target_2=reward_2,
                risk_reward_target_1=rr_1,
                risk_reward_target_2=rr_2,
            )

        # --------------------------------------------------------------
        # Position sizing
        # --------------------------------------------------------------

        position_size = None
        capital_required = None
        maximum_risk_amount = None

        if account_size is not None:
            (
                position_size,
                capital_required,
                maximum_risk_amount,
            ) = self._calculate_position_size(
                account_size=account_size,
                entry=entry,
                risk_per_share=risk_per_share,
            )

        # --------------------------------------------------------------
        # Expected return
        # --------------------------------------------------------------

        expected_return = self._calculate_expected_return(
            inputs=inputs,
            entry=entry,
            target=target_2,
        )

        # --------------------------------------------------------------
        # Quality score
        # --------------------------------------------------------------

        quality_score = self._calculate_quality_score(
            confidence=inputs.confidence,
            risk_reward=primary_rr,
            predicted_range_available=(
                lower_price is not None
                and upper_price is not None
            ),
        )

        if inputs.lower_return is None or inputs.upper_return is None:
            warnings.append(
                "Predicted price range is incomplete."
            )

        if inputs.atr is None:
            warnings.append(
                "ATR was not available; stop-loss uses percentage logic."
            )

        return SwingSetup(
            status=SetupStatus.VALID,
            direction=inputs.direction,
            entry_price=entry,
            stop_loss=stop_loss,
            target_1=target_1,
            target_2=target_2,
            predicted_lower_price=lower_price,
            predicted_median_price=median_price,
            predicted_upper_price=upper_price,
            risk_per_share=risk_per_share,
            reward_to_target_1=reward_1,
            reward_to_target_2=reward_2,
            risk_reward_target_1=rr_1,
            risk_reward_target_2=rr_2,
            position_size=position_size,
            capital_required=capital_required,
            maximum_risk_amount=maximum_risk_amount,
            expected_return=expected_return,
            confidence=inputs.confidence,
            quality_score=quality_score,
            reason="Swing setup passed construction and risk checks.",
            warnings=warnings,
            metadata=dict(inputs.metadata),
        )

    # ------------------------------------------------------------------
    # Stop loss
    # ------------------------------------------------------------------

    def _calculate_stop_loss(
        self,
        inputs: SwingSetupInput,
        entry: float,
    ) -> Optional[float]:
        """
        Calculate stop loss.

        Priority:
            1. Support/resistance
            2. ATR
            3. Percentage fallback

        The stop is always constrained by configured minimum/maximum
        percentage distance.
        """

        min_distance = (
            entry * self.config.minimum_stop_loss_pct
        )

        max_distance = (
            entry * self.config.maximum_stop_loss_pct
        )

        default_distance = (
            entry * self.config.default_stop_loss_pct
        )

        if inputs.direction == SetupDirection.LONG:
            if inputs.support_price is not None:
                raw_stop = inputs.support_price

            elif inputs.atr is not None:
                raw_stop = entry - (2.0 * inputs.atr)

            else:
                raw_stop = entry - default_distance

            distance = entry - raw_stop

            distance = max(distance, min_distance)
            distance = min(distance, max_distance)

            stop = entry - distance

            if stop >= entry:
                return None

            return stop

        if inputs.direction == SetupDirection.SHORT:
            if inputs.resistance_price is not None:
                raw_stop = inputs.resistance_price

            elif inputs.atr is not None:
                raw_stop = entry + (2.0 * inputs.atr)

            else:
                raw_stop = entry + default_distance

            distance = raw_stop - entry

            distance = max(distance, min_distance)
            distance = min(distance, max_distance)

            stop = entry + distance

            if stop <= entry:
                return None

            return stop

        return None

    # ------------------------------------------------------------------
    # Targets
    # ------------------------------------------------------------------

    def _calculate_targets(
        self,
        inputs: SwingSetupInput,
        entry: float,
        stop_loss: float,
        predicted_median_price: Optional[float],
        predicted_upper_price: Optional[float],
        predicted_lower_price: Optional[float],
    ) -> tuple[float, float]:
        """
        Calculate Target 1 and Target 2.

        The model's predicted range is preferred.

        If a sufficient predicted range is unavailable, targets are
        constructed using the minimum configured risk/reward ratio.
        """

        risk = abs(entry - stop_loss)

        minimum_reward = (
            risk * self.config.minimum_risk_reward
        )

        if inputs.direction == SetupDirection.LONG:
            minimum_target = entry + minimum_reward

            # Target 2 prefers the model's upper prediction.
            if predicted_upper_price is not None:
                target_2 = max(
                    predicted_upper_price,
                    minimum_target,
                )
            elif predicted_median_price is not None:
                target_2 = max(
                    predicted_median_price,
                    minimum_target,
                )
            else:
                target_2 = minimum_target

            # Target 1 is a partial-profit level.
            if predicted_median_price is not None:
                target_1 = max(
                    predicted_median_price,
                    entry + (
                        risk * self.config.minimum_risk_reward
                        * self.config.target_1_fraction
                    ),
                )
            else:
                target_1 = entry + (
                    risk
                    * self.config.minimum_risk_reward
                    * self.config.target_1_fraction
                )

            # Ensure ordering.
            target_1 = min(target_1, target_2)

            return target_1, target_2

        if inputs.direction == SetupDirection.SHORT:
            minimum_target = entry - minimum_reward

            if predicted_lower_price is not None:
                target_2 = min(
                    predicted_lower_price,
                    minimum_target,
                )
            elif predicted_median_price is not None:
                target_2 = min(
                    predicted_median_price,
                    minimum_target,
                )
            else:
                target_2 = minimum_target

            if predicted_median_price is not None:
                target_1 = min(
                    predicted_median_price,
                    entry - (
                        risk * self.config.minimum_risk_reward
                        * self.config.target_1_fraction
                    ),
                )
            else:
                target_1 = entry - (
                    risk
                    * self.config.minimum_risk_reward
                    * self.config.target_1_fraction
                )

            target_1 = max(target_1, target_2)

            return target_1, target_2

        return entry, entry

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------

    def _calculate_position_size(
        self,
        account_size: float,
        entry: float,
        risk_per_share: float,
    ) -> tuple[int, float, float]:
        """
        Calculate risk-based position size.

        Position size is constrained by both:
            - maximum account risk
            - maximum capital allocation
        """

        maximum_risk_amount = (
            account_size * self.config.max_risk_per_trade
        )

        if self.config.use_risk_based_position_sizing:
            risk_based_quantity = math.floor(
                maximum_risk_amount / risk_per_share
            )
        else:
            risk_based_quantity = 1

        maximum_capital = (
            account_size
            * self.config.maximum_capital_fraction
        )

        capital_based_quantity = math.floor(
            maximum_capital / entry
        )

        position_size = min(
            risk_based_quantity,
            capital_based_quantity,
        )

        position_size = max(position_size, 0)

        capital_required = (
            position_size * entry
        )

        return (
            int(position_size),
            float(capital_required),
            float(maximum_risk_amount),
        )

    # ------------------------------------------------------------------
    # Utility calculations
    # ------------------------------------------------------------------

    @staticmethod
    def _return_to_price(
        current_price: float,
        return_value: Optional[float],
    ) -> Optional[float]:
        if return_value is None:
            return None

        if not math.isfinite(return_value):
            return None

        return float(
            current_price * (1.0 + return_value)
        )

    @staticmethod
    def _reward(
        direction: SetupDirection,
        entry: float,
        target: float,
    ) -> float:
        if direction == SetupDirection.LONG:
            return max(0.0, target - entry)

        if direction == SetupDirection.SHORT:
            return max(0.0, entry - target)

        return 0.0

    @staticmethod
    def _safe_divide(
        numerator: float,
        denominator: float,
    ) -> Optional[float]:
        if denominator <= 0:
            return None

        return float(numerator / denominator)

    @staticmethod
    def _calculate_expected_return(
        inputs: SwingSetupInput,
        entry: float,
        target: float,
    ) -> Optional[float]:
        """
        Calculate expected return from the target price.

        This is a simple target-based estimate and is not a probability-
        weighted expected value.
        """

        if entry <= 0:
            return None

        if inputs.direction == SetupDirection.LONG:
            return float((target / entry) - 1.0)

        if inputs.direction == SetupDirection.SHORT:
            return float((entry / target) - 1.0)

        return None

    @staticmethod
    def _calculate_quality_score(
        confidence: float,
        risk_reward: float,
        predicted_range_available: bool,
    ) -> float:
        """
        Calculate a diagnostic setup-quality score.

        This score is NOT a probability of profit.
        """

        confidence_component = min(
            max(confidence, 0.0),
            1.0,
        )

        rr_component = min(
            max(risk_reward / 4.0, 0.0),
            1.0,
        )

        range_component = (
            1.0 if predicted_range_available else 0.5
        )

        score = (
            0.50 * confidence_component
            + 0.35 * rr_component
            + 0.15 * range_component
        )

        return float(min(max(score, 0.0), 1.0))

    # ------------------------------------------------------------------
    # Status helpers
    # ------------------------------------------------------------------

    def _invalid_setup(
        self,
        inputs: SwingSetupInput,
        reason: str,
    ) -> SwingSetup:
        return SwingSetup(
            status=SetupStatus.INVALID,
            direction=inputs.direction,
            entry_price=None,
            stop_loss=None,
            target_1=None,
            target_2=None,
            predicted_lower_price=None,
            predicted_median_price=None,
            predicted_upper_price=None,
            risk_per_share=None,
            reward_to_target_1=None,
            reward_to_target_2=None,
            risk_reward_target_1=None,
            risk_reward_target_2=None,
            position_size=None,
            capital_required=None,
            maximum_risk_amount=None,
            expected_return=None,
            confidence=inputs.confidence,
            quality_score=0.0,
            reason=reason,
            warnings=[],
            metadata=dict(inputs.metadata),
        )

    def _no_trade_setup(
        self,
        inputs: SwingSetupInput,
        reason: str,
        **calculated: object,
    ) -> SwingSetup:
        return SwingSetup(
            status=SetupStatus.NO_TRADE,
            direction=inputs.direction,
            entry_price=calculated.get("entry_price"),
            stop_loss=calculated.get("stop_loss"),
            target_1=calculated.get("target_1"),
            target_2=calculated.get("target_2"),
            predicted_lower_price=calculated.get(
                "predicted_lower_price"
            ),
            predicted_median_price=calculated.get(
                "predicted_median_price"
            ),
            predicted_upper_price=calculated.get(
                "predicted_upper_price"
            ),
            risk_per_share=calculated.get(
                "risk_per_share"
            ),
            reward_to_target_1=calculated.get(
                "reward_to_target_1"
            ),
            reward_to_target_2=calculated.get(
                "reward_to_target_2"
            ),
            risk_reward_target_1=calculated.get(
                "risk_reward_target_1"
            ),
            risk_reward_target_2=calculated.get(
                "risk_reward_target_2"
            ),
            position_size=None,
            capital_required=None,
            maximum_risk_amount=None,
            expected_return=None,
            confidence=inputs.confidence,
            quality_score=0.0,
            reason=reason,
            warnings=[],
            metadata=dict(inputs.metadata),
        )


def setup_to_dict(
    setup: SwingSetup,
) -> Dict[str, object]:
    """Convert a SwingSetup into a dashboard/API-friendly dictionary."""

    return {
        "status": setup.status.value,
        "direction": setup.direction.value,
        "entry_price": setup.entry_price,
        "stop_loss": setup.stop_loss,
        "target_1": setup.target_1,
        "target_2": setup.target_2,
        "predicted_lower_price": setup.predicted_lower_price,
        "predicted_median_price": setup.predicted_median_price,
        "predicted_upper_price": setup.predicted_upper_price,
        "risk_per_share": setup.risk_per_share,
        "reward_to_target_1": setup.reward_to_target_1,
        "reward_to_target_2": setup.reward_to_target_2,
        "risk_reward_target_1": setup.risk_reward_target_1,
        "risk_reward_target_2": setup.risk_reward_target_2,
        "position_size": setup.position_size,
        "capital_required": setup.capital_required,
        "maximum_risk_amount": setup.maximum_risk_amount,
        "expected_return": setup.expected_return,
        "confidence": setup.confidence,
        "quality_score": setup.quality_score,
        "reason": setup.reason,
        "warnings": list(setup.warnings),
        "metadata": dict(setup.metadata),
    }


def setup_summary(
    setup: SwingSetup,
) -> Dict[str, object]:
    """Return a compact summary suitable for Streamlit."""

    return {
        "status": setup.status.value,
        "direction": setup.direction.value,
        "entry": setup.entry_price,
        "stop_loss": setup.stop_loss,
        "target_1": setup.target_1,
        "target_2": setup.target_2,
        "risk_reward": setup.risk_reward_target_2,
        "position_size": setup.position_size,
        "capital_required": setup.capital_required,
        "confidence": setup.confidence,
        "quality_score": setup.quality_score,
    }


__all__ = [
    "SetupDirection",
    "SetupStatus",
    "SwingSetupConfig",
    "SwingSetupInput",
    "SwingSetup",
    "SwingSetupEngine",
    "setup_to_dict",
    "setup_summary",
]
