from __future__ import annotations

from dataclasses import dataclass
import sys
from typing import Iterable

import numpy as np
import pandas as pd

from .debug import Debug


@dataclass
class SelectionResult:
    df: pd.DataFrame
    selection_suffix: str = ""
    selection_value_tag: str | None = None
    pair_mask: np.ndarray | None = None


class ParticipantSelector:
    operator_tokens = {"==": "eq", "!=": "ne", ">": "gt", "<": "lt", ">=": "ge", "<=": "le"}

    def __init__(self, debug: Debug | None = None, exit_on_error: bool = True) -> None:
        self.debug = debug or Debug()
        self.exit_on_error = exit_on_error

    @staticmethod
    def _sanitize_token(token) -> str:
        token = str(token).strip()
        return token.replace(" ", "-").replace("/", "-").replace(",", "-").replace("|", "-")

    @staticmethod
    def _normalize_identifier(value) -> str:
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="ignore")
        if pd.isna(value):
            return ""
        return str(value).strip()

    def _fail(self, message: str) -> None:
        if self.debug:
            self.debug.error(message)
        if self.exit_on_error:
            sys.exit(1)
        raise ValueError(message)

    def apply(self,
              df: pd.DataFrame,
              selections: Iterable[str] | None,
              subject_ids: Iterable | None = None,
              session_ids: Iterable | None = None) -> SelectionResult:
        selections = list(selections or [])
        selection_suffix = ""
        selection_value_tag = None
        pair_mask = None

        if subject_ids is not None or session_ids is not None:
            if subject_ids is None or session_ids is None:
                self._fail("subject_ids and session_ids must both be provided when filtering pairs.")
            subject_ids = list(subject_ids)
            session_ids = list(session_ids)
            if len(subject_ids) != len(session_ids):
                self._fail("subject_ids and session_ids must have the same length.")

        if not selections:
            if subject_ids is not None:
                pair_mask = np.ones(len(subject_ids), dtype=bool)
            return SelectionResult(df=df, selection_suffix=selection_suffix,
                                   selection_value_tag=selection_value_tag, pair_mask=pair_mask)

        covariate_names = df.columns.to_list()
        covariate_map = {str(col).lower(): col for col in covariate_names}
        covar_conditions: dict[str, list[tuple[str, str, str]]] = {}
        for raw_select in selections:
            try:
                covar_name, condition_raw = [part.strip() for part in raw_select.split(",", 1)]
            except ValueError:
                self._fail(
                    f"Invalid --select format '{raw_select}'. Use COVARNAME,VALUE (e.g., --select Diag,1)"
                )

            covar_key = covar_name
            if covar_name not in covariate_names:
                covar_key = covariate_map.get(covar_name.lower())
            if covar_key is None:
                self._fail(
                    f"Covariate '{covar_name}' not found in participants file. Available covariates: {covariate_names}"
                )

            operator = "=="
            value_str = condition_raw
            for candidate_op in [">=", "<=", "!=", ">", "<"]:
                if condition_raw.startswith(candidate_op):
                    operator = candidate_op
                    value_str = condition_raw[len(candidate_op):].strip()
                    break
            else:
                if condition_raw.startswith("=="):
                    operator = "=="
                    value_str = condition_raw[2:].strip()

            if value_str == "":
                self._fail(f"Missing comparison value in selection '{raw_select}'.")

            covar_conditions.setdefault(covar_key, []).append((operator, value_str, raw_select))

        mask = pd.Series(True, index=df.index)
        selection_suffix_parts: list[str] = []
        eq_values_by_covar: dict[str, list[str]] = {}
        other_conditions_by_covar: dict[str, list[tuple[str, object, str, pd.Series]]] = {}

        for covar_name, conditions in covar_conditions.items():
            covar_series = df[covar_name]
            is_numeric = pd.api.types.is_numeric_dtype(covar_series)
            covar_series_numeric = None
            try:
                numeric_candidate = pd.to_numeric(covar_series, errors="coerce")
            except Exception:
                numeric_candidate = None
            if is_numeric:
                covar_series_numeric = numeric_candidate
            elif numeric_candidate is not None:
                nonmissing_mask = covar_series.notna()
                if bool(nonmissing_mask.any()) and bool(numeric_candidate[nonmissing_mask].notna().all()):
                    is_numeric = True
                    covar_series_numeric = numeric_candidate
            if is_numeric and covar_series_numeric is None:
                self._fail(f"Covariate '{covar_name}' must be numeric for comparisons.")

            eq_values: list[object] = []
            eq_value_strs: list[str] = []
            other_conditions: list[tuple[str, object, str, pd.Series]] = []
            desc_parts: list[str] = []

            for operator, value_str, raw_select in conditions:
                needs_numeric = operator in [">", "<", ">=", "<="]
                if needs_numeric:
                    if covar_series_numeric is None:
                        self._fail(
                            f"Covariate '{covar_name}' must be numeric for comparison with '{operator}'."
                        )
                    try:
                        covar_value = pd.to_numeric(value_str)
                    except ValueError:
                        self._fail(
                            f"Covariate '{covar_name}' expects numeric values; could not parse '{value_str}'."
                        )
                    compare_series = covar_series_numeric
                else:
                    if is_numeric:
                        try:
                            covar_value = pd.to_numeric(value_str)
                        except ValueError:
                            self._fail(
                                f"Covariate '{covar_name}' expects numeric values; could not parse '{value_str}'."
                            )
                        compare_series = covar_series_numeric
                    else:
                        if operator not in ["==", "!="]:
                            self._fail(
                                f"Covariate '{covar_name}' is non-numeric; only == or != comparisons are supported."
                            )
                        covar_value = value_str
                        compare_series = covar_series

                if operator == "==":
                    eq_values.append(covar_value)
                    eq_value_strs.append(value_str)
                elif operator in {"!=", ">", "<", ">=", "<="}:
                    other_conditions.append((operator, covar_value, value_str, compare_series))
                else:
                    self._fail(f"Unsupported operator '{operator}' in selection '{raw_select}'.")

            covar_mask = pd.Series(True, index=df.index)
            covar_name_tag = covar_name.replace(" ", "-")
            if eq_values:
                eq_series = covar_series_numeric if covar_series_numeric is not None else covar_series
                covar_mask &= eq_series.isin(eq_values)
                desc_parts.append(f"in {eq_values}")
                in_values = "_".join(self._sanitize_token(v) for v in eq_value_strs)
                selection_suffix_parts.append(f"{covar_name_tag}-in{in_values}")

            for operator, covar_value, value_str, compare_series in other_conditions:
                if operator == "!=":
                    covar_mask &= compare_series != covar_value
                elif operator == ">":
                    covar_mask &= compare_series > covar_value
                elif operator == "<":
                    covar_mask &= compare_series < covar_value
                elif operator == ">=":
                    covar_mask &= compare_series >= covar_value
                elif operator == "<=":
                    covar_mask &= compare_series <= covar_value
                desc_parts.append(f"{operator} {covar_value}")
                selection_suffix_parts.append(
                    f"{covar_name_tag}-{self.operator_tokens.get(operator, operator)}{self._sanitize_token(value_str)}"
                )

            mask &= covar_mask
            if desc_parts:
                self.debug.info(
                    f"Applying participant filter {covar_name} {' and '.join(desc_parts)}; {mask.sum()} rows remain"
                )

            eq_values_by_covar[covar_name] = eq_value_strs
            other_conditions_by_covar[covar_name] = other_conditions

        df = df[mask]
        if df.empty:
            self._fail(f"No participants matched filters: {selections}")

        if len(covar_conditions) == 1:
            only_covar = next(iter(covar_conditions))
            only_eq_values = eq_values_by_covar.get(only_covar, [])
            only_other = other_conditions_by_covar.get(only_covar, [])
            if len(only_eq_values) == 1 and not only_other:
                selection_suffix = f"sel-{only_covar.replace(' ', '-')}"
                selection_value_tag = self._sanitize_token(only_eq_values[0])
            else:
                selection_suffix = "sel-" + "_".join(selection_suffix_parts)
        else:
            selection_suffix = "sel-" + "_".join(selection_suffix_parts)

        if subject_ids is not None:
            if "participant_id" not in df.columns or "session_id" not in df.columns:
                self._fail("Participants DataFrame must contain participant_id and session_id columns for pair filtering.")
            selected_pairs = {
                (
                    self._normalize_identifier(row.participant_id),
                    self._normalize_identifier(row.session_id),
                )
                for row in df.itertuples(index=False)
            }
            current_pairs = [
                (
                    self._normalize_identifier(subj),
                    self._normalize_identifier(sess),
                )
                for subj, sess in zip(subject_ids, session_ids)
            ]
            pair_mask = np.array([pair in selected_pairs for pair in current_pairs], dtype=bool)
            if pair_mask.sum() == 0:
                self._fail("No matching subject/session pairs between selection filters and dataset.")

        return SelectionResult(
            df=df,
            selection_suffix=selection_suffix,
            selection_value_tag=selection_value_tag,
            pair_mask=pair_mask,
        )
