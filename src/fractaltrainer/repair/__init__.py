from fractaltrainer.repair.repair_loop import RepairLoop, RepairAttempt
from fractaltrainer.repair.context import (
    GeometricContextGatherer,
    GeometricRepairContext,
    ALLOWED_FILES,
)
from fractaltrainer.repair.hparam_config import (
    HPARAM_SCHEMA,
    load_hparams,
    save_hparams,
    validate_hparams,
)
from fractaltrainer.repair.patch_parser import PatchParser, CodePatch, ParseResult
from fractaltrainer.repair.prompt_builder import PromptBuilder
from fractaltrainer.repair.llm_client import (
    make_claude_cli_client,
    make_claude_client,
    make_local_llm_client,
)

__all__ = [
    "RepairLoop",
    "RepairAttempt",
    "GeometricContextGatherer",
    "GeometricRepairContext",
    "ALLOWED_FILES",
    "HPARAM_SCHEMA",
    "load_hparams",
    "save_hparams",
    "validate_hparams",
    "PatchParser",
    "CodePatch",
    "ParseResult",
    "PromptBuilder",
    "make_claude_cli_client",
    "make_claude_client",
    "make_local_llm_client",
]