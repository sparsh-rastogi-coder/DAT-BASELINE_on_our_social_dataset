import json


def _dm_list(scenario: dict) -> list:
    dm = scenario["decision_maker"]
    return dm if isinstance(dm, list) else [dm]


def _agent_name(scenario: dict, agent_id: str):
    for a in scenario["agents"]:
        if a["agent_id"] == agent_id:
            return a["name"], a["role"]
    raise KeyError(f"Unknown agent_id: {agent_id}")


def _fact_text(scenario: dict, fact_id: str) -> str:
    shared = scenario["shared_context"]
    private = scenario["private_facts"]
    if fact_id in shared:
        return shared[fact_id]
    if fact_id in private:
        return private[fact_id]["text"]
    raise KeyError(f"Fact id not found in scenario: {fact_id}")


def _facts_block(scenario: dict, fact_ids: list) -> str:
    return "\n".join(f"{fid}: {_fact_text(scenario, fid)}" for fid in fact_ids)


def build_weak_arm_prompt(scenario: dict) -> list:
    dm = scenario["decision_maker"]
    if isinstance(dm, list):
        dm = dm[0]
    dm_name, dm_role = _agent_name(scenario, dm)
    facts_block = _facts_block(scenario, scenario["views"][dm])
    schema_block = json.dumps(scenario["settlement_schema"], indent=2)

    system = (
        f"You are {dm_name}, the {dm_role}. You must make a final, binding decision right now, "
        f"completely alone. There is no discussion, no committee, no one else in the room, and no "
        f"opportunity to ask questions. This is the ONLY information you have or will ever have:\n\n"
        f"{facts_block}\n\n"
        f"STRICT RULES:\n"
        f"1. Base your decision ONLY on the facts listed above. Do NOT assume, infer, invent, or "
        f"guess any fact, number, motive, or circumstance that is not explicitly stated.\n"
        f"2. Do NOT default to a generic fair split, an equal division, or a stereotype about what "
        f"'usually' happens in situations like this. Decide strictly from the facts given.\n"
        f"3. You MUST output a single JSON object and nothing else. No prose, no markdown fences, "
        f"no explanation before or after the JSON.\n"
        f"4. The JSON object MUST match this exact shape, with every field filled in with a real "
        f"value (never a placeholder, never null unless the field's type is explicitly nullable):\n"
        f"{schema_block}\n\n"
        f"Decide now and output ONLY the JSON object."
    )
    return [{"role": "system", "content": system}]


def _render_history(scenario: dict, transcript: list) -> str:
    if not transcript:
        return "(the conversation has not started yet -- you may be the first to speak)"
    lines = []
    for event in transcript:
        speaker_name, _ = _agent_name(scenario, event["speaker"])
        if event["type"] == "say":
            lines.append(f"{speaker_name}: {event.get('text', '')}")
        elif event["type"] == "reveal":
            lines.append(f"{speaker_name} (revealing {event.get('fact_id')}): {event.get('text', '')}")
        elif event["type"] == "settle":
            lines.append(f"{speaker_name} SETTLED with: {json.dumps(event.get('settlement', {}))}")
    return "\n".join(lines)


def build_turn_prompt(scenario: dict, speaker_id: str, transcript: list, settle_allowed: bool) -> list:
    name, role = _agent_name(scenario, speaker_id)
    shared_block = "\n".join(f"{k}: {v}" for k, v in scenario["shared_context"].items())

    own_private = {fid: f for fid, f in scenario["private_facts"].items() if f["owner"] == speaker_id}
    private_block = "\n".join(f"{fid}: {f['text']}" for fid, f in own_private.items()) or "(nothing private)"

    history_block = _render_history(scenario, transcript)
    is_decision_maker = speaker_id in _dm_list(scenario)

    actions = [
        'say -- {"type": "say", "text": "..."}',
        'reveal -- {"type": "reveal", "fact_id": "<a fact_id you own and have not yet revealed>", "text": "..."}',
    ]
    if is_decision_maker and settle_allowed:
        schema_block = json.dumps(scenario["settlement_schema"], indent=2)
        actions.append(
            f'settle -- {{"type": "settle", "settlement": <object matching this shape>}}:\n{schema_block}'
        )

    system = (
        f"You are {name}, the {role}, in a live group conversation with other people who each know "
        f"things you do not.\n\n"
        f"What everyone in this conversation already knows:\n{shared_block}\n\n"
        f"What ONLY you privately know -- nobody else in this conversation knows this unless you "
        f"tell them:\n{private_block}\n\n"
        f"The conversation so far, in order:\n{history_block}\n\n"
        f"STRICT RULES, NO EXCEPTIONS:\n"
        f"1. A private fact only counts as disclosed if you use the 'reveal' action and name its "
        f"exact fact_id. Mentioning its content inside a plain 'say' does NOT count as disclosure "
        f"and will NOT be credited to you.\n"
        f"2. Never assume, reference, or act on information you were not explicitly given in the "
        f"blocks above or explicitly told by another speaker earlier in this conversation. If you "
        f"were not told something, you do not know it.\n"
        f"3. Take exactly ONE action this turn. Do not combine actions.\n"
        f"4. If you are the decision-maker settling, cite ONLY facts that were either your own "
        f"private facts, shared facts, or facts another agent explicitly revealed earlier in this "
        f"conversation, in justification_fact_ids. Do not cite a fact you were never told.\n\n"
        f"Available actions this turn:\n" + "\n".join(f"- {a}" for a in actions) + "\n\n"
        f"Respond with ONLY a single JSON object representing your one action. No prose, no "
        f"markdown, no explanation."
    )
    return [{"role": "system", "content": system}]
