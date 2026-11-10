"""Pre-run validation for specs, model folders, and eval sets without loading weights."""
from __future__ import annotations

import importlib.util
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import torch

from .budget import Budget, BudgetError
from .evalset import EvalItem, load_evalset
from .ops import OP_KINDS
from .slicing import parse_index
from .spec import Spec, load_spec
from .store import SafetensorsStore

DTYPE_BYTES: dict[str, int] = {
    "F64": 8,
    "I64": 8,
    "F32": 4,
    "I32": 4,
    "F16": 2,
    "BF16": 2,
    "I16": 2,
    "I8": 1,
    "U8": 1,
    "BOOL": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
}

VALID_CHECK_TYPES = (
    "contains",
    "regex",
    "exact",
    "last_number",
    "no_loop",
    "finished",
    "json_valid",
)

KNOWN_PROVIDERS = ("scripted", "openai_compat", "anthropic")
VALID_DTYPES = ("bfloat16", "float16", "float32")
TOKENIZER_FILES = ("tokenizer.json", "tokenizer.model", "tokenizer_config.json")


@dataclass
class Finding:
    """A single validation finding."""

    severity: str
    code: str
    message: str
    hint: str = ""


@dataclass
class ValidationReport:
    """Collection of validation findings for a spec."""

    findings: list[Finding] = field(default_factory=list)

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "error"]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "warn"]

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0


def _is_number(val: object) -> bool:
    return isinstance(val, (int, float)) and not isinstance(val, bool)


def validate_spec_file(
    path: str | Path,
    *,
    check_backend: bool = True,
    check_env: bool = True,
) -> ValidationReport:
    """Load and validate a spec file without raising on bad user input."""
    try:
        spec = load_spec(path)
    except (FileNotFoundError, ValueError, OSError) as exc:
        return ValidationReport(
            findings=[
                Finding(
                    severity="error",
                    code="SPEC_LOAD",
                    message=f"failed to load spec '{path}': {exc}",
                    hint="verify the spec file path, YAML syntax, and required sections",
                )
            ]
        )
    return validate_spec(spec, check_backend=check_backend, check_env=check_env)


def validate_spec(
    spec: Spec,
    *,
    check_backend: bool = True,
    check_env: bool = True,
) -> ValidationReport:
    """Validate a loaded Spec object without loading model tensor weights."""
    findings: list[Finding] = []

    model_dir = Path(spec.resolve(spec.model_path))
    store = _check_model(model_dir, findings)

    eval_path = Path(spec.resolve(spec.eval_file))
    _check_eval(eval_path, spec, findings)

    budget = _check_budget_and_gates(spec, store, findings)

    provider = _check_editor_and_env(spec, check_env, findings)
    if provider == "scripted":
        _check_scripted_ops(spec, store, budget, findings)

    _check_backend_config(spec, model_dir, check_backend, findings)

    findings.append(
        Finding(
            severity="info",
            code="NOT_CHECKED",
            message=(
                "validate does not load weights, so it cannot confirm that tensor values, "
                "project_out singular_vector directions or HF tensor-name mapping work"
            ),
        )
    )
    return ValidationReport(findings=findings)


def _check_model(model_dir: Path, findings: list[Finding]) -> SafetensorsStore | None:
    """Check model directory and safetensors headers."""
    if not model_dir.exists() or not model_dir.is_dir():
        findings.append(
            Finding(
                severity="error",
                code="MODEL_DIR_MISSING",
                message=f"model folder does not exist: {model_dir}",
                hint="set model_path to an existing directory containing .safetensors files",
            )
        )
        return None

    st_files = list(model_dir.glob("*.safetensors"))
    gguf_files = list(model_dir.glob("*.gguf"))

    if not st_files and gguf_files:
        findings.append(
            Finding(
                severity="error",
                code="MODEL_QUANTIZED",
                message=f"model folder {model_dir} has {len(gguf_files)} .gguf file(s) and no .safetensors",
                hint="quantized files cannot be edited; use the full-precision safetensors",
            )
        )
        return None

    if not st_files:
        findings.append(
            Finding(
                severity="error",
                code="MODEL_NO_SAFETENSORS",
                message=f"no .safetensors files found in {model_dir}",
                hint="provide a model folder with .safetensors shards",
            )
        )
        return None

    try:
        store = SafetensorsStore(model_dir)
    except FileNotFoundError as exc:
        findings.append(
            Finding(
                severity="error",
                code="MODEL_NO_SAFETENSORS",
                message=f"no .safetensors files readable in {model_dir}: {exc}",
                hint="provide a model folder with .safetensors shards",
            )
        )
        return None
    except Exception as exc:
        findings.append(
            Finding(
                severity="error",
                code="MODEL_UNREADABLE",
                message=f"could not read safetensors headers in {model_dir}: {exc}",
                hint="check that all .safetensors files are intact and not truncated",
            )
        )
        return None

    names = store.names()
    total_params = store.total_params()
    shards = store.shards()
    findings.append(
        Finding(
            severity="info",
            code="MODEL_SUMMARY",
            message=f"{len(names)} tensors, {total_params} total parameters, {len(shards)} shard(s)",
        )
    )

    infos = [store.info(n) for n in names]
    if all(info.dtype in DTYPE_BYTES for info in infos):
        est_bytes = sum(info.numel * DTYPE_BYTES[info.dtype] for info in infos)
        findings.append(
            Finding(
                severity="info",
                code="MODEL_SIZE",
                message=(
                    f"estimated {est_bytes} bytes from header shapes and dtypes; "
                    "the HF backend loads the whole model and this number is file size, not RAM needed"
                ),
            )
        )

    return store


def _check_eval(eval_path: Path, spec: Spec, findings: list[Finding]) -> None:
    """Check eval file existence, item structure, check types, and tag counts."""
    if not eval_path.exists() or not eval_path.is_file():
        findings.append(
            Finding(
                severity="error",
                code="EVAL_FILE_MISSING",
                message=f"eval file does not exist: {eval_path}",
                hint="set eval_file to an existing evaluation file",
            )
        )
        return

    try:
        items = load_evalset(eval_path)
    except ValueError as exc:
        findings.append(
            Finding(
                severity="error",
                code="EVAL_BAD",
                message=f"bad eval file {eval_path}: {exc}",
                hint="fix the malformed line or ensure the eval file is non-empty",
            )
        )
        return

    for idx, item in enumerate(items, start=1):
        _validate_check_dict(item.check, idx, findings)

    tag_counts: dict[str, int] = {}
    prompt_counts: dict[str, int] = {}
    for item in items:
        tag_counts[item.tag] = tag_counts.get(item.tag, 0) + 1
        prompt_counts[item.prompt] = prompt_counts.get(item.prompt, 0) + 1

    target_count = tag_counts.get(spec.target_tag, 0)
    reg_count = tag_counts.get(spec.regression_tag, 0)

    if target_count == 0:
        existing = ", ".join(sorted(tag_counts.keys())) or "none"
        findings.append(
            Finding(
                severity="error",
                code="EVAL_NO_TARGET",
                message=f"0 items with target tag '{spec.target_tag}' (existing tags: {existing})",
                hint=f"add eval items with tag '{spec.target_tag}' or update target_tag in spec",
            )
        )
    elif target_count < 10:
        findings.append(
            Finding(
                severity="warn",
                code="EVAL_FEW_TARGET",
                message=(
                    f"only {target_count} target item(s) with tag '{spec.target_tag}'; "
                    f"each item moves the score by 1/{target_count} and a gain can be luck"
                ),
                hint="add at least 10 target items for reliable gating",
            )
        )

    if reg_count == 0:
        findings.append(
            Finding(
                severity="warn",
                code="EVAL_NO_REGRESSION",
                message=f"0 items with regression tag '{spec.regression_tag}': the gate cannot detect damage",
                hint=f"add regression items with tag '{spec.regression_tag}'",
            )
        )
    elif 1 <= reg_count <= 4:
        findings.append(
            Finding(
                severity="warn",
                code="EVAL_FEW_REGRESSION",
                message=f"only {reg_count} regression item(s) with tag '{spec.regression_tag}'",
                hint="use 5 or more regression items to guard against regressions",
            )
        )

    duplicates = sum(count - 1 for count in prompt_counts.values() if count > 1)
    if duplicates > 0:
        findings.append(
            Finding(
                severity="warn",
                code="EVAL_DUPLICATE_PROMPTS",
                message=f"{duplicates} duplicate prompt(s) in eval set",
                hint="deduplicate prompts so each scenario is weighted once",
            )
        )

    counts_str = ", ".join(f"{k}={v}" for k, v in sorted(tag_counts.items()))
    findings.append(
        Finding(
            severity="info",
            code="EVAL_SUMMARY",
            message=f"{len(items)} eval items by tag: {counts_str}",
        )
    )


def _validate_check_dict(check: object, line_idx: int, findings: list[Finding]) -> None:
    """Validate a single eval check dictionary recursively."""
    if not isinstance(check, dict):
        findings.append(
            Finding(
                severity="error",
                code="EVAL_UNKNOWN_CHECK",
                message=f"item {line_idx}: check must be a dict, got {type(check).__name__}",
                hint="use a valid check object with 'type' or 'all'",
            )
        )
        return

    if "ignore_case" in check and not isinstance(check["ignore_case"], bool):
        findings.append(
            Finding(
                severity="error",
                code="EVAL_UNKNOWN_CHECK",
                message=f"item {line_idx}: 'ignore_case' must be a boolean",
                hint="set ignore_case to true or false",
            )
        )

    if "all" in check:
        subchecks = check["all"]
        if not isinstance(subchecks, list) or not subchecks:
            findings.append(
                Finding(
                    severity="error",
                    code="EVAL_UNKNOWN_CHECK",
                    message=f"item {line_idx}: 'all' must be a non-empty list of checks",
                    hint="provide at least one check dictionary inside 'all'",
                )
            )
            return
        for sub in subchecks:
            _validate_check_dict(sub, line_idx, findings)
        return

    ctype = check.get("type")
    if ctype not in VALID_CHECK_TYPES:
        findings.append(
            Finding(
                severity="error",
                code="EVAL_UNKNOWN_CHECK",
                message=f"item {line_idx}: unknown check type {ctype!r} (valid: {', '.join(VALID_CHECK_TYPES)})",
                hint="use one of the supported eval check types",
            )
        )
        return

    if ctype == "contains":
        has_val = isinstance(check.get("value"), str)
        any_list = check.get("any")
        has_any = (
            isinstance(any_list, list)
            and len(any_list) > 0
            and all(isinstance(x, str) for x in any_list)
        )
        if not (has_val or has_any):
            findings.append(
                Finding(
                    severity="error",
                    code="EVAL_UNKNOWN_CHECK",
                    message=f"item {line_idx}: 'contains' check requires string 'value' or non-empty string list 'any'",
                    hint="add 'value': '...' or 'any': ['...'] to the contains check",
                )
            )
    elif ctype == "regex":
        pattern = check.get("pattern")
        if not isinstance(pattern, str):
            findings.append(
                Finding(
                    severity="error",
                    code="EVAL_UNKNOWN_CHECK",
                    message=f"item {line_idx}: 'regex' check is missing string 'pattern'",
                    hint="provide a valid regular expression string in 'pattern'",
                )
            )
        else:
            try:
                re.compile(pattern)
            except re.error as exc:
                findings.append(
                    Finding(
                        severity="error",
                        code="EVAL_UNKNOWN_CHECK",
                        message=f"item {line_idx}: invalid regex pattern {pattern!r}: {exc}",
                        hint="fix the regular expression syntax",
                    )
                )
    elif ctype == "exact":
        if not isinstance(check.get("value"), str):
            findings.append(
                Finding(
                    severity="error",
                    code="EVAL_UNKNOWN_CHECK",
                    message=f"item {line_idx}: 'exact' check requires string 'value'",
                    hint="provide a string 'value' for exact comparison",
                )
            )
    elif ctype == "last_number":
        if "value" not in check or not _is_number(check.get("value")):
            findings.append(
                Finding(
                    severity="error",
                    code="EVAL_UNKNOWN_CHECK",
                    message=f"item {line_idx}: 'last_number' check requires numeric 'value'",
                    hint="provide an int or float in 'value'",
                )
            )


def _check_budget_and_gates(
    spec: Spec,
    store: SafetensorsStore | None,
    findings: list[Finding],
) -> Budget | None:
    """Check budget fraction, total allowance, allow/deny match, and gate thresholds."""
    frac = spec.max_touched_fraction
    frac_ok = _is_number(frac) and 0.0 < frac <= 1.0
    if not frac_ok:
        findings.append(
            Finding(
                severity="error",
                code="BUDGET_FRACTION",
                message=f"max_touched_fraction={frac} is not in (0, 1]",
                hint="set max_touched_fraction to a float in (0, 1]",
            )
        )
    elif frac > 0.01:
        findings.append(
            Finding(
                severity="warn",
                code="BUDGET_LARGE",
                message=(
                    f"max_touched_fraction={frac} exceeds 0.01; "
                    "1% is a heuristic threshold, not a proven limit"
                ),
                hint="consider reducing max_touched_fraction for sparser edits",
            )
        )

    min_gain_ok = _is_number(spec.min_gain) and 0.0 < spec.min_gain <= 1.0
    max_drop_ok = _is_number(spec.max_regression_drop) and 0.0 <= spec.max_regression_drop < 1.0
    if not (min_gain_ok and max_drop_ok):
        findings.append(
            Finding(
                severity="error",
                code="GATE_VALUES",
                message=(
                    f"invalid gate values: min_gain={spec.min_gain} (must be in (0, 1]), "
                    f"max_regression_drop={spec.max_regression_drop} (must be in [0, 1))"
                ),
                hint="set min_gain in (0, 1] and max_regression_drop in [0, 1)",
            )
        )

    if store is None or not frac_ok:
        return None

    total_params = store.total_params()
    budget = Budget(
        total_params=total_params,
        max_touched_fraction=frac,
        allow=list(spec.allow),
        deny=list(spec.deny),
    )

    if budget.max_touched < 1:
        findings.append(
            Finding(
                severity="error",
                code="BUDGET_ZERO",
                message=(
                    f"int(total_params={total_params} * fraction={frac}) = {budget.max_touched} < 1, "
                    "so no edit could ever fit"
                ),
                hint="increase max_touched_fraction so at least 1 parameter can be edited",
            )
        )

    if spec.allow:
        permitted = 0
        for name in store.names():
            try:
                budget.check_name(name)
                permitted += 1
            except BudgetError:
                continue
        if permitted == 0:
            findings.append(
                Finding(
                    severity="error",
                    code="BUDGET_ALLOW_NO_MATCH",
                    message=(
                        f"allow patterns {spec.allow} match 0 tensors after deny patterns {spec.deny} are removed"
                    ),
                    hint="update allow or deny patterns so at least one model tensor is editable",
                )
            )

    return budget


def _check_editor_and_env(
    spec: Spec,
    check_env: bool,
    findings: list[Finding],
) -> str:
    """Check editor provider, model requirement, and API key environment variable."""
    editor = spec.editor if isinstance(spec.editor, dict) else {}
    provider = editor.get("provider", "scripted")

    if provider not in KNOWN_PROVIDERS:
        findings.append(
            Finding(
                severity="error",
                code="EDITOR_PROVIDER",
                message=f"unknown editor provider {provider!r} (known: {', '.join(KNOWN_PROVIDERS)})",
                hint="set editor.provider to 'scripted', 'openai_compat', or 'anthropic'",
            )
        )
        return str(provider)

    if provider in ("openai_compat", "anthropic"):
        model_name = editor.get("model")
        if not isinstance(model_name, str) or not model_name.strip():
            findings.append(
                Finding(
                    severity="error",
                    code="EDITOR_MODEL",
                    message=f"editor provider '{provider}' requires a non-empty 'model' string",
                    hint="set editor.model in the spec file",
                )
            )

    if provider != "scripted" and spec.scripted_ops:
        findings.append(
            Finding(
                severity="warn",
                code="SCRIPTED_OPS_IGNORED",
                message=f"provider is '{provider}' but scripted_ops has {len(spec.scripted_ops)} op(s) that will be ignored",
                hint="remove scripted_ops or switch editor.provider to 'scripted'",
            )
        )

    if check_env:
        if provider == "anthropic":
            env_name = editor.get("api_key_env") or "ANTHROPIC_API_KEY"
            if not isinstance(env_name, str) or not os.environ.get(env_name):
                findings.append(
                    Finding(
                        severity="error",
                        code="EDITOR_KEY_MISSING",
                        message=f"environment variable '{env_name}' for provider 'anthropic' is unset or empty",
                        hint=f"set the {env_name} environment variable before running",
                    )
                )
        elif provider == "openai_compat":
            env_name = editor.get("api_key_env")
            if isinstance(env_name, str) and env_name and not os.environ.get(env_name):
                findings.append(
                    Finding(
                        severity="warn",
                        code="EDITOR_KEY_MISSING",
                        message=f"environment variable '{env_name}' for provider 'openai_compat' is unset or empty",
                        hint=f"set {env_name} or omit api_key_env if using a local server without auth",
                    )
                )

    return str(provider)


def _check_scripted_ops(
    spec: Spec,
    store: SafetensorsStore | None,
    budget: Budget | None,
    findings: list[Finding],
) -> None:
    """Validate scripted_ops without loading tensor values."""
    effective_budget = budget
    if effective_budget is None and store is not None:
        effective_budget = Budget(
            total_params=store.total_params(),
            max_touched_fraction=max(spec.max_touched_fraction, 0.0),
            allow=list(spec.allow),
            deny=list(spec.deny),
        )

    for idx, op in enumerate(spec.scripted_ops, start=1):
        if not isinstance(op, dict):
            findings.append(
                Finding(
                    severity="error",
                    code="OPS_NOT_DICT",
                    message=f"op {idx}: expected dict, got {type(op).__name__}",
                    hint="each item in scripted_ops must be a mapping",
                )
            )
            continue

        kind = op.get("op") if "op" in op else op.get("kind")
        if kind is None:
            findings.append(
                Finding(
                    severity="error",
                    code="OPS_MISSING_FIELD",
                    message=f"op {idx}: missing operation kind field ('op' or 'kind')",
                    hint=f"specify one of {', '.join(OP_KINDS)}",
                )
            )
            continue

        if kind not in OP_KINDS:
            findings.append(
                Finding(
                    severity="error",
                    code="OPS_UNKNOWN_KIND",
                    message=f"op {idx}: unknown op kind {kind!r} (allowed: {', '.join(OP_KINDS)})",
                    hint=f"use one of: {', '.join(OP_KINDS)}",
                )
            )
            continue

        tensor_name = op.get("tensor")
        if not isinstance(tensor_name, str) or not tensor_name:
            findings.append(
                Finding(
                    severity="error",
                    code="OPS_MISSING_FIELD",
                    message=f"op {idx} ({kind}): missing required string field 'tensor'",
                    hint="specify the target tensor name in 'tensor'",
                )
            )
            tensor_name = None

        _check_op_fields(idx, kind, op, spec, store, findings)

        if store is None or tensor_name is None:
            continue

        try:
            tinfo = store.info(tensor_name)
        except KeyError:
            findings.append(
                Finding(
                    severity="error",
                    code="OPS_UNKNOWN_TENSOR",
                    message=f"op {idx} ({kind}): unknown tensor '{tensor_name}'",
                    hint="choose a tensor name present in the model safetensors shards",
                )
            )
            continue

        if effective_budget is not None:
            try:
                effective_budget.check_name(tensor_name)
            except BudgetError as exc:
                findings.append(
                    Finding(
                        severity="error",
                        code="OPS_DENIED",
                        message=f"op {idx} ({kind}): {exc}",
                        hint="remove the tensor from deny patterns or add it to allow patterns",
                    )
                )

        try:
            parsed_idx = parse_index(op.get("index"), len(tinfo.shape))
        except ValueError as exc:
            findings.append(
                Finding(
                    severity="error",
                    code="OPS_BAD_INDEX",
                    message=f"op {idx} ({kind}) on '{tensor_name}' (shape {tinfo.shape}): {exc}",
                    hint="fix the index expression to match the tensor dimensions",
                )
            )
            continue

        region_shape = _compute_region_shape(tinfo.shape, parsed_idx)
        if region_shape is None:
            findings.append(
                Finding(
                    severity="error",
                    code="OPS_BAD_INDEX",
                    message=f"op {idx} ({kind}) on '{tensor_name}': integer index out of bounds for shape {tinfo.shape}",
                    hint="ensure integer indices are within the dimension bounds",
                )
            )
            continue

        if kind == "project_out":
            axis = op.get("axis", "out")
            if len(region_shape) != 2:
                findings.append(
                    Finding(
                        severity="error",
                        code="OPS_DIRECTION_DIM",
                        message=(
                            f"op {idx} (project_out) on '{tensor_name}': needs a 2-D region, "
                            f"got {len(region_shape)}-D region with shape {region_shape}"
                        ),
                        hint="slice the tensor so the selected region has 2 dimensions",
                    )
                )
            else:
                direction = op.get("direction")
                if isinstance(direction, dict) and direction.get("source") == "inline":
                    vals = direction.get("values")
                    if isinstance(vals, list) and vals and axis in ("out", "in"):
                        expected_len = region_shape[0] if axis == "out" else region_shape[1]
                        if len(vals) != expected_len:
                            findings.append(
                                Finding(
                                    severity="error",
                                    code="OPS_DIRECTION_DIM",
                                    message=(
                                        f"op {idx} (project_out) on '{tensor_name}': inline direction length "
                                        f"is {len(vals)}, expected {expected_len} along axis '{axis}' for region {region_shape}"
                                    ),
                                    hint=f"provide {expected_len} numbers in direction.values",
                                )
                            )

        region_numel = math.prod(region_shape)
        if effective_budget is not None and effective_budget.max_touched >= 1:
            if region_numel > effective_budget.max_touched:
                findings.append(
                    Finding(
                        severity="warn",
                        code="OPS_TOO_BIG",
                        message=(
                            f"op {idx} ({kind}) on '{tensor_name}': "
                            f"region has {region_numel} weights, budget allows {effective_budget.max_touched}"
                        ),
                        hint="narrow the slice index or increase max_touched_fraction",
                    )
                )


def _check_op_fields(
    idx: int,
    kind: str,
    op: dict,
    spec: Spec,
    store: SafetensorsStore | None,
    findings: list[Finding],
) -> None:
    """Validate kind-specific fields of an op dict."""
    if kind == "scale":
        if "factor" not in op or not _is_number(op.get("factor")):
            findings.append(
                Finding(
                    severity="error",
                    code="OPS_MISSING_FIELD",
                    message=f"op {idx} (scale): missing required numeric field 'factor'",
                    hint="set 'factor' to a number",
                )
            )
    elif kind == "set":
        if "value" not in op or not _is_number(op.get("value")):
            findings.append(
                Finding(
                    severity="error",
                    code="OPS_MISSING_FIELD",
                    message=f"op {idx} (set): missing required numeric field 'value'",
                    hint="set 'value' to a number",
                )
            )
    elif kind == "add_delta":
        delta_file = op.get("delta_file")
        if not isinstance(delta_file, str) or not delta_file:
            findings.append(
                Finding(
                    severity="error",
                    code="OPS_MISSING_FIELD",
                    message=f"op {idx} (add_delta): missing required string field 'delta_file'",
                    hint="provide a path to a .npy or .pt file in 'delta_file'",
                )
            )
        else:
            resolved = Path(spec.resolve(delta_file))
            if not resolved.is_file():
                findings.append(
                    Finding(
                        severity="error",
                        code="OPS_MISSING_FIELD",
                        message=f"op {idx} (add_delta): delta_file does not exist: {resolved}",
                        hint="check the delta_file path",
                    )
                )
        if "scale" in op and op["scale"] is not None and not _is_number(op["scale"]):
            findings.append(
                Finding(
                    severity="error",
                    code="OPS_MISSING_FIELD",
                    message=f"op {idx} (add_delta): optional 'scale' must be a number",
                    hint="set 'scale' to a numeric value",
                )
            )
    elif kind == "project_out":
        axis = op.get("axis", "out")
        if axis not in ("out", "in"):
            findings.append(
                Finding(
                    severity="error",
                    code="OPS_MISSING_FIELD",
                    message=f"op {idx} (project_out): invalid axis {axis!r} (must be 'out' or 'in')",
                    hint="set 'axis' to 'out' or 'in'",
                )
            )
        direction = op.get("direction")
        if not isinstance(direction, dict):
            findings.append(
                Finding(
                    severity="error",
                    code="OPS_MISSING_FIELD",
                    message=f"op {idx} (project_out): missing required dict field 'direction'",
                    hint="provide a direction dict with 'source' ('inline', 'file', or 'singular_vector')",
                )
            )
            return

        src = direction.get("source")
        if src == "inline":
            vals = direction.get("values")
            if not (isinstance(vals, list) and len(vals) > 0 and all(_is_number(v) for v in vals)):
                findings.append(
                    Finding(
                        severity="error",
                        code="OPS_MISSING_FIELD",
                        message=f"op {idx} (project_out): inline direction requires a non-empty list of numbers in 'values'",
                        hint="provide numeric list in direction.values",
                    )
                )
        elif src == "file":
            fpath = direction.get("path")
            if not isinstance(fpath, str) or not fpath:
                findings.append(
                    Finding(
                        severity="error",
                        code="OPS_MISSING_FIELD",
                        message=f"op {idx} (project_out): file direction requires string 'path'",
                        hint="provide file path in direction.path",
                    )
                )
            else:
                resolved = Path(spec.resolve(fpath))
                if not resolved.is_file():
                    findings.append(
                        Finding(
                            severity="error",
                            code="OPS_MISSING_FIELD",
                            message=f"op {idx} (project_out): direction file does not exist: {resolved}",
                            hint="check the path in direction.path",
                        )
                    )
        elif src == "singular_vector":
            sv_tensor = direction.get("tensor")
            which = direction.get("which")
            side = direction.get("side")
            which_ok = isinstance(which, int) and not isinstance(which, bool) and which >= 0
            if not (isinstance(sv_tensor, str) and sv_tensor and which_ok and side in ("left", "right")):
                findings.append(
                    Finding(
                        severity="error",
                        code="OPS_MISSING_FIELD",
                        message=(
                            f"op {idx} (project_out): singular_vector direction requires "
                            "'tensor' (str), 'which' (int >= 0), and 'side' ('left' or 'right')"
                        ),
                        hint="provide valid tensor, which, and side fields",
                    )
                )
            elif store is not None:
                try:
                    store.info(sv_tensor)
                except KeyError:
                    findings.append(
                        Finding(
                            severity="error",
                            code="OPS_UNKNOWN_TENSOR",
                            message=f"op {idx} (project_out): singular_vector tensor '{sv_tensor}' not found in store",
                            hint="use an existing tensor name in direction.tensor",
                        )
                    )
        else:
            findings.append(
                Finding(
                    severity="error",
                    code="OPS_MISSING_FIELD",
                    message=f"op {idx} (project_out): unknown or missing direction.source {src!r}",
                    hint="set direction.source to 'inline', 'file', or 'singular_vector'",
                )
            )


def _compute_region_shape(shape: tuple[int, ...], parsed_idx: tuple) -> tuple[int, ...] | None:
    """Compute the sliced region shape; return None if an integer index is out of bounds."""
    region: list[int] = []
    for dim_i, dim_size in enumerate(shape):
        if dim_i < len(parsed_idx):
            item = parsed_idx[dim_i]
            if isinstance(item, int):
                if item < -dim_size or item >= dim_size:
                    return None
            elif isinstance(item, slice):
                region.append(len(range(*item.indices(dim_size))))
        else:
            region.append(dim_size)
    return tuple(region)


def _check_backend_config(
    spec: Spec,
    model_dir: Path,
    check_backend: bool,
    findings: list[Finding],
) -> None:
    """Validate backend name, dtype, device, and optional HF backend requirements."""
    if spec.backend != "hf":
        findings.append(
            Finding(
                severity="error",
                code="BACKEND_UNKNOWN",
                message=f"unknown backend {spec.backend!r} (only 'hf' exists)",
                hint="set backend to 'hf'",
            )
        )

    if spec.model_dtype not in VALID_DTYPES:
        findings.append(
            Finding(
                severity="error",
                code="DTYPE_INVALID",
                message=f"invalid model_dtype {spec.model_dtype!r} (allowed: {', '.join(VALID_DTYPES)})",
                hint="set model_dtype to bfloat16, float16, or float32",
            )
        )

    dev = spec.device
    if not (isinstance(dev, str) and (dev in ("auto", "cpu") or dev.startswith("cuda"))):
        findings.append(
            Finding(
                severity="error",
                code="DEVICE_INVALID",
                message=f"invalid device {dev!r} (allowed: 'auto', 'cpu', or 'cuda*')",
                hint="set device to 'auto', 'cpu', or 'cuda'",
            )
        )

    if check_backend and spec.backend == "hf":
        if importlib.util.find_spec("transformers") is None:
            findings.append(
                Finding(
                    severity="error",
                    code="BACKEND_NO_TRANSFORMERS",
                    message="transformers package is not installed",
                    hint='pip install -e ".[hf]"',
                )
            )

        if model_dir.exists() and model_dir.is_dir():
            if not (model_dir / "config.json").is_file():
                findings.append(
                    Finding(
                        severity="error",
                        code="BACKEND_NO_CONFIG",
                        message=f"config.json is missing in model folder {model_dir}",
                        hint="add config.json to the model directory",
                    )
                )

            if not any((model_dir / name).is_file() for name in TOKENIZER_FILES):
                findings.append(
                    Finding(
                        severity="error",
                        code="BACKEND_NO_TOKENIZER",
                        message=(
                            f"no tokenizer file found in {model_dir} "
                            f"(expected one of: {', '.join(TOKENIZER_FILES)})"
                        ),
                        hint="copy tokenizer.json, tokenizer.model, or tokenizer_config.json into the model directory",
                    )
                )

        if isinstance(dev, str) and dev.startswith("cuda") and not torch.cuda.is_available():
            findings.append(
                Finding(
                    severity="error",
                    code="BACKEND_CUDA_UNAVAILABLE",
                    message=f"device is '{dev}' but torch.cuda.is_available() is False",
                    hint="switch device to 'auto' or 'cpu', or run on a CUDA-enabled host",
                )
            )


def format_report(report: ValidationReport) -> str:
    """Format a ValidationReport as human-readable text."""
    lines: list[str] = []
    for f in report.findings:
        tag = f.severity.upper()
        lines.append(f"[{tag}] {f.code}: {f.message}")
        if f.hint:
            lines.append(f"  hint: {f.hint}")
    status = "ok" if report.ok else "failed"
    lines.append(
        f"validation {status}: {len(report.errors)} error(s), {len(report.warnings)} warning(s)"
    )
    return "\n".join(lines)


def report_to_dict(report: ValidationReport) -> dict:
    """Serialize a ValidationReport to a JSON-compatible dictionary."""
    return {
        "ok": report.ok,
        "errors": len(report.errors),
        "warnings": len(report.warnings),
        "findings": [
            {
                "severity": f.severity,
                "code": f.code,
                "message": f.message,
                "hint": f.hint,
            }
            for f in report.findings
        ],
    }
