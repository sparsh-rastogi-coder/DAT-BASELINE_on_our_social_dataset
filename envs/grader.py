import ast

ALLOWED_NAMES = {"decisions", "credited_facts", "commitments", "justification_fact_ids", "revealed"}
ALLOWED_BUILTINS = {"any": any, "all": all, "len": len, "sum": sum}
ALLOWED_METHODS = {"values", "keys", "items", "get"}


def validate_check(check_str: str) -> ast.AST:
    try:
        tree = ast.parse(check_str, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"Check is not valid Python syntax: {e} in {check_str!r}")

    bound_names = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            for generator in node.generators:
                for target_node in ast.walk(generator.target):
                    if isinstance(target_node, ast.Name):
                        bound_names.add(target_node.id)

    allowed = ALLOWED_NAMES | bound_names

    allowed_attribute_nodes = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in ALLOWED_METHODS:
                allowed_attribute_nodes.add(node.func)

    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id not in allowed and node.id not in ALLOWED_BUILTINS:
            raise ValueError(f"Check uses disallowed name: {node.id!r} in {check_str!r}")
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id not in ALLOWED_BUILTINS:
                raise ValueError(f"Check calls disallowed function: {node.func.id!r} in {check_str!r}")
            if isinstance(node.func, ast.Attribute) and node.func not in allowed_attribute_nodes:
                raise ValueError(f"Check calls disallowed method: .{node.func.attr}() in {check_str!r}")
        if isinstance(node, ast.Attribute) and node not in allowed_attribute_nodes:
            raise ValueError(f"Disallowed attribute access: .{node.attr} in {check_str!r}")
        if isinstance(node, ast.IfExp):
            raise ValueError(f"Ternary expressions are not allowed: {check_str!r}")
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.Lambda)):
            raise ValueError(f"Disallowed construct in check: {check_str!r}")
    return tree


def validate_scenario_checks(scenario: dict):
    for block in ("content_checks", "provenance_checks"):
        for check_id, check_str in scenario.get(block, {}).items():
            validate_check(check_str)


def eval_check(check_str: str, decisions, credited_facts, commitments, justification_fact_ids, revealed) -> bool:
    validate_check(check_str)
    namespace = {
        "decisions": decisions,
        "credited_facts": credited_facts,
        "commitments": commitments,
        "justification_fact_ids": justification_fact_ids,
        "revealed": revealed,
    }
    return bool(eval(check_str, {"__builtins__": ALLOWED_BUILTINS}, namespace))


def build_revealed_set(transcript: list) -> list:
    return [event["fact_id"] for event in transcript if event.get("type") == "reveal"]


def grade_content(scenario: dict, settlement: dict) -> dict:
    results = {}
    for check_id, check_str in scenario.get("content_checks", {}).items():
        try:
            results[check_id] = eval_check(
                check_str,
                decisions=settlement.get("decisions", {}),
                credited_facts=settlement.get("credited_facts", []),
                commitments=settlement.get("commitments", []),
                justification_fact_ids=settlement.get("justification_fact_ids", []),
                revealed=[],
            )
        except Exception:
            results[check_id] = False
    return results


def grade_provenance(scenario: dict, settlement: dict, transcript: list) -> dict:
    revealed = build_revealed_set(transcript)
    results = {}
    for check_id, check_str in scenario.get("provenance_checks", {}).items():
        try:
            results[check_id] = eval_check(
                check_str,
                decisions=settlement.get("decisions", {}),
                credited_facts=settlement.get("credited_facts", []),
                commitments=settlement.get("commitments", []),
                justification_fact_ids=settlement.get("justification_fact_ids", []),
                revealed=revealed,
            )
        except Exception:
            results[check_id] = False
    return results