"""MATH answer grading adapted from OpenAI PRM800K/Hendrycks normalization."""

from __future__ import annotations

import re

import sympy
from pylatexenc import latex2text
from sympy.parsing import sympy_parser

TUPLE_CHARS = "()[]"
BAD_SUBSTRINGS = ("^{", "^(")
BAD_REGEXES = (r"\^[0-9]+\^", r"\^[0-9][0-9]+")


def extract_math_answer(text: str) -> str | None:
    final_matches: list[str] = re.findall(
        r"FINAL_ANSWER:\s*(.+)", text, flags=re.IGNORECASE
    )
    if final_matches:
        return final_matches[-1].strip().rstrip(".")
    marker = text.rfind(r"\boxed")
    if marker < 0:
        return None
    start = text.find("{", marker)
    if start < 0:
        return None
    depth = 0
    for index in range(start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : index].strip()
    return None


def grade_math_answer(given_answer: str | None, reference_solution: str) -> bool:
    ground_truth = extract_math_answer(reference_solution)
    if given_answer is None or ground_truth is None:
        return False
    normalized_gold = normalize_math_answer(ground_truth)
    normalized_given = normalize_math_answer(given_answer)
    if normalized_gold == normalized_given:
        return True
    if not normalized_gold or not normalized_given:
        return False

    gold_elements = split_tuple(_normalize_for_sympy(ground_truth))
    given_elements = split_tuple(_normalize_for_sympy(given_answer))
    if len(gold_elements) != len(given_elements):
        return False
    if len(gold_elements) > 1 and (
        normalized_gold[0] != normalized_given[0]
        or normalized_gold[-1] != normalized_given[-1]
    ):
        return False
    for gold, given in zip(gold_elements, given_elements, strict=True):
        if _is_fraction(gold) and _is_fraction(given):
            if gold != given:
                return False
        elif _string_is_int(gold) != _string_is_int(given):
            return False
        elif not _equal_under_sympy(gold, given):
            return False
    return True


def normalize_math_answer(answer: str | None) -> str | None:
    if answer is None:
        return None
    answer = answer.strip()
    try:
        match = re.search(r"^\\text\{(?P<text>.+?)\}$", answer)
        if match is not None:
            answer = match.group("text").strip()
        return _strip_math_string(answer)
    except (AssertionError, IndexError):
        return answer


def _strip_math_string(value: str) -> str:
    value = value.replace("\n", "")
    value = value.replace(r"\!", "")
    value = value.replace(r"\\", "\\")
    value = value.replace("tfrac", "frac").replace("dfrac", "frac")
    value = value.replace(r"\left", "").replace(r"\right", "")
    value = value.replace(r"^{\circ}", "").replace(r"^\circ", "")
    value = value.replace(r"\$", "")
    if r"\text{ " in value:
        value = value.split(r"\text{ ", maxsplit=1)[0]
    value = value.replace(r"\%", "").replace(r"\%", "")
    value = value.replace(" .", " 0.").replace("{.", "{0.")
    if not value:
        return value
    if value[0] == ".":
        value = "0" + value
    if len(value.split("=")) == 2 and len(value.split("=")[0]) <= 2:
        value = value.split("=")[1]
    value = _fix_sqrt(value)
    value = value.replace(" ", "")
    value = _fix_fracs(value)
    if value == "0.5":
        value = r"\frac{1}{2}"
    return _fix_simple_slash(value)


def _fix_fracs(value: str) -> str:
    pieces = value.split(r"\frac")
    rebuilt = pieces[0]
    for piece in pieces[1:]:
        rebuilt += r"\frac"
        if piece.startswith("{"):
            rebuilt += piece
            continue
        if len(piece) < 2:
            return value
        numerator, denominator = piece[0], piece[1]
        rest = piece[2:]
        if denominator != "{":
            rebuilt += f"{{{numerator}}}{{{denominator}}}{rest}"
        else:
            rebuilt += f"{{{numerator}}}{denominator}{rest}"
    return rebuilt


def _fix_simple_slash(value: str) -> str:
    if len(value.split("/")) != 2:
        return value
    numerator, denominator = value.split("/")
    try:
        return rf"\frac{{{int(numerator)}}}{{{int(denominator)}}}"
    except ValueError:
        return value


def _fix_sqrt(value: str) -> str:
    if r"\sqrt" not in value:
        return value
    pieces = value.split(r"\sqrt")
    rebuilt = pieces[0]
    for piece in pieces[1:]:
        if not piece:
            return value
        if piece[0] != "{":
            rebuilt += rf"\sqrt{{{piece[0]}}}{piece[1:]}"
        else:
            rebuilt += r"\sqrt" + piece
    return rebuilt


def _normalize_for_sympy(expression: str) -> str:
    expression = expression.strip()
    match = re.search(r"^\\text\{(?P<text>.+?)\}$", expression)
    if match is not None:
        expression = match.group("text")
    expression = expression.replace(r"\%", "%").replace(r"\$", "$")
    expression = expression.replace("$", "").replace("%", "")
    expression = expression.replace(" or ", " , ").replace(" and ", " , ")
    expression = expression.replace("million", "*10^6")
    expression = expression.replace("billion", "*10^9")
    expression = expression.replace("trillion", "*10^12")
    for unit in (
        "degree",
        "cm",
        "centimeter",
        "meter",
        "mile",
        "second",
        "minute",
        "hour",
        "day",
        "week",
        "month",
        "year",
        "foot",
        "feet",
        "inch",
        "yard",
    ):
        expression = re.sub(rf"{unit}(es)?(s)? *(\^[0-9]+)?", "", expression)
    expression = re.sub(r"\^ *\\circ", "", expression)
    if expression.startswith("{") and expression.endswith("}"):
        expression = expression[1:-1]
    expression = re.sub(r",\\! *", "", expression)
    if _is_float(expression) and _is_int(float(expression)):
        expression = str(round(float(expression)))
    if "\\" in expression:
        try:
            expression = _parse_latex(expression)
        except (TypeError, ValueError):
            pass
    expression = re.sub("- *", "-", expression)
    expression = re.sub(r"([0-9]) +([0-9])", r"\1+\2", expression)
    expression = expression.replace(" ", "").replace("{", "").replace("}", "")
    expression = expression.lower()
    if _string_is_int(expression):
        expression = str(int(float(expression.replace(",", ""))))
    return expression


def _parse_latex(expression: str) -> str:
    expression = expression.replace(r"\tfrac", r"\frac")
    expression = expression.replace(r"\dfrac", r"\frac")
    expression = expression.replace(r"\frac", r" \frac")
    expression = latex2text.LatexNodes2Text().latex_to_text(expression)
    replacements = {
        "√": "sqrt",
        "π": "pi",
        "∞": "inf",
        "∪": "U",  # noqa: RUF001
        "·": "*",
        "×": "*",  # noqa: RUF001
    }
    for source, target in replacements.items():
        expression = expression.replace(source, target)
    return expression.strip()


def _equal_under_sympy(gold: str, given: str) -> bool:
    try:
        difference = f"({gold})-({given})"
        if not _allow_sympy(difference):
            return False
        parsed = sympy_parser.parse_expr(
            difference.replace("^", "**"),
            transformations=(
                *sympy_parser.standard_transformations,
                sympy_parser.implicit_multiplication_application,
            ),
        )
        return bool(sympy.simplify(parsed) == 0)
    except (TypeError, ValueError, SyntaxError):
        return False


def _allow_sympy(expression: str) -> bool:
    letters = set(
        expression.replace("sqrt", "").replace("frac", "")
    ) & set("abcdefghijklmnopqrstuvwxyz")
    return (
        len(letters) <= 2
        and not any(value in expression for value in BAD_SUBSTRINGS)
        and not any(re.search(pattern, expression) for pattern in BAD_REGEXES)
    )


def split_tuple(expression: str) -> list[str]:
    expression = _strip_commas(expression)
    if not expression:
        return []
    if (
        len(expression) > 2
        and expression[0] in TUPLE_CHARS
        and expression[-1] in TUPLE_CHARS
        and all(character not in expression[1:-1] for character in TUPLE_CHARS)
    ):
        return [element.strip() for element in expression[1:-1].split(",")]
    return [expression]


def _strip_commas(expression: str) -> str:
    pattern = re.compile(r"(\d)(,)(\d\d\d)($|\D)")
    while True:
        next_expression = pattern.sub(r"\1\3\4", expression)
        if next_expression == expression:
            return next_expression
        expression = next_expression


def _is_float(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


def _is_int(value: float) -> bool:
    return abs(value - round(value)) <= 1e-7


def _string_is_int(value: str) -> bool:
    try:
        return _is_int(float(_strip_commas(value)))
    except ValueError:
        return False


def _is_fraction(expression: str) -> bool:
    return bool(re.search(r"^-?[0-9]+.?/0*[1-9][0-9]*.?$", expression))
