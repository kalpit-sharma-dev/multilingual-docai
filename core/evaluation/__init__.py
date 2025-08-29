"""
Evaluation Module

Provides evaluation tools for the PS-05 system including:
- Layout detection evaluation
- OCR evaluation
- Language identification evaluation
- Natural language generation evaluation
"""

from .layout_evaluator import LayoutEvaluator, evaluate_layout
from .ocr_evaluator import OCREvaluator, evaluate_ocr
from .langid_evaluator import LanguageEvaluator, evaluate_language_id
from .nl_evaluator import NLEvaluator, evaluate_nl_generation

__all__ = [
    "LayoutEvaluator",
    "OCREvaluator", 
    "LanguageEvaluator",
    "NLEvaluator",
    "evaluate_layout",
    "evaluate_ocr",
    "evaluate_language_id",
    "evaluate_nl_generation"
] 