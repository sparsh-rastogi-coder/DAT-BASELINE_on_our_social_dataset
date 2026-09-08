"""
EducationEnv: A DAT environment for the Coevolving Social Agents education scenarios.

Wraps the 100 education scenarios in final_scenarios.json with the same
reset(i) / step(state, action, prefixes) interface used by SotopiaEnv,
so td3.py can use it without modification.

Reward function: deterministic grader using content_checks + provenance_checks.
No GPT-4 required.
"""

import os
import sys
import json
import re
import pickle
import time
import numpy as np
import torch
from typing import Literal, Optional

# ─── path resolution so this works from both dat/ and envs/ ──────────────────
current_file_path = os.path.abspath(__file__)
ENVS_DIR = os.path.dirname(current_file_path)
REPO_ROOT = os.path.dirname(ENVS_DIR)
DAT_DIR = os.path.join(REPO_ROOT, "dat")
PIPELINE_DIR = os.path.join(REPO_ROOT, "Coevolving_Social_Agents_Pipeline")

# grader.py and prompt_builder.py are co-located in envs/ (committed to git)
sys.path.insert(0, ENVS_DIR)
# ControlLLM lives in dat/
sys.path.insert(0, DAT_DIR)

from ControlLLM import ControlLLM
from grader import grade_content, grade_provenance
from prompt_builder import build_turn_prompt

SCENARIOS_PATH = os.path.join(ENVS_DIR, "final_scenarios.json")


# ─── helpers ──────────────────────────────────────────────────────────────────

def _extract_json(text: str) -> dict:
    """Extract the first valid JSON object from raw LLM output."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    # Fallback: treat raw text as a plain 'say' action
    return {"type": "say", "text": text[:500]}


def _compute_reward(scenario: dict, transcript: list) -> float:
    """
    Score the episode by checking the settlement against all grader checks.
    Returns fraction of checks passed in [0.0, 1.0].
    Returns 0.0 if no 'settle' event found in transcript.
    """
    settlement = None
    for event in transcript:
        if event.get("type") == "settle":
            settlement = event.get("settlement", {})
            break
    if settlement is None:
        return 0.0

    content_results = grade_content(scenario, settlement)
    provenance_results = grade_provenance(scenario, settlement, transcript)

    all_checks = list(content_results.values()) + list(provenance_results.values())
    if not all_checks:
        return 0.0
    score = float(sum(all_checks)) / len(all_checks)
    print(f"  Grader: content={content_results}, provenance={provenance_results}, score={score:.3f}")
    return score


# ─── environment ──────────────────────────────────────────────────────────────

class EducationEnv:
    """
    DAT environment wrapping 100 education scenarios.

    Interface matches SotopiaEnv / RedTeamEnv so td3.py works unchanged:
        state = env.reset(i)
        next_state, reward, done, info = env.step(state, action, prefixes)
    """

    def __init__(
        self,
        model_name: str,
        env_model: str = "gpt-4",           # unused – we use deterministic grader
        opponent_model: str = "",            # unused – single model shared
        prefix_size: int = 2,
        prefix_embedding_size: int = 64,
        max_turns: int = 10,
        temperature: float = 0.7,
        prefix_pos: Literal['start', 'mid', 'end'] = 'start',
        judge_temp: float = 10.0,           # unused – kept for API compat
        test_baseline: bool = False,
        test_gpt: bool = False,              # unused
        saving_dir: Optional[str] = None,
        mode: str = "train",                 # unused – kept for API compat
        hf_token: Optional[str] = None,
    ):
        print(f"[EduEnv] Loading ControlLLM({model_name}) …")
        self.model = ControlLLM(
            model_name,
            prefix_size,
            prefix_embedding_size,
            prefix_pos,
            hf_token=hf_token,
        )
        self.prefix_size = prefix_size
        self.temperature = temperature
        self.max_turns = max_turns
        self.test_baseline = test_baseline

        print(f"[EduEnv] Loading scenarios from {SCENARIOS_PATH} …")
        with open(SCENARIOS_PATH, 'r', encoding='utf-8') as f:
            self.scenarios = json.load(f)
        self.queries = self.scenarios  # alias used by eval_actor's len() check

        self.saving_dir = saving_dir
        if saving_dir:
            os.makedirs(saving_dir, exist_ok=True)

        # Episode state – initialised by reset()
        self.cur_scenario: Optional[dict] = None
        self.transcript: list = []
        self.global_turn: int = 0
        self.controlled_agent_id: Optional[str] = None
        self.turn_order: list = []
        self.turn_cap: int = 0

        # Transition buffers (mirrors SotopiaEnv)
        self.cur_states: list = []
        self.next_states: list = []
        self.actions: list = []
        self.cur_rewards: list = []
        self.terminations: list = []

    # ── prompt helpers ────────────────────────────────────────────────────────

    def _build_prompt_str(self, agent_id: str, settle_allowed: bool = True) -> str:
        """Build a flat string prompt for agent_id at the current transcript."""
        messages = build_turn_prompt(
            self.cur_scenario, agent_id, self.transcript, settle_allowed
        )
        parts = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "system":
                parts.append(content)
            else:
                parts.append(f"{role.capitalize()}: {content}")
        return "\n\n".join(parts)

    def _get_input_tensors(self, prompt: str):
        toks = self.model.tokenizer(
            prompt, return_tensors='pt',
            truncation=True, max_length=800
        )
        return toks['input_ids'].cuda(), toks['attention_mask'].cuda()

    # ── state ─────────────────────────────────────────────────────────────────

    def get_state(self) -> torch.Tensor:
        """Return the 4096-dim LLM hidden state for the decision_maker's current prompt."""
        prompt = self._build_prompt_str(self.controlled_agent_id, settle_allowed=True)
        input_ids, attention_mask = self._get_input_tensors(prompt)
        dev = self.model.base_model.device
        input_ids = input_ids.to(dev)
        attention_mask = attention_mask.to(dev)

        with torch.no_grad():
            output = self.model.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
        last_hidden = output.hidden_states[-1][:, -1, :].to(torch.float32)
        return last_hidden  # shape (1, 4096)

    # ── episode ────────────────────────────────────────────────────────────────

    def reset(self, scenario_idx: int = -1) -> torch.Tensor:
        """Load scenario `scenario_idx` (random if -1) and return initial state."""
        if scenario_idx < 0 or scenario_idx >= len(self.scenarios):
            scenario_idx = int(np.random.randint(0, len(self.scenarios)))

        self.cur_scenario = self.scenarios[scenario_idx]
        ic = self.cur_scenario["interaction_config"]
        self.turn_order = ic["turn_order"]
        self.turn_cap = min(ic.get("turn_cap", self.max_turns), self.max_turns)

        dm = self.cur_scenario["decision_maker"]
        self.controlled_agent_id = dm if isinstance(dm, str) else dm[0]

        self.transcript = []
        self.global_turn = 0

        self.cur_states = []
        self.next_states = []
        self.actions = []
        self.cur_rewards = []
        self.terminations = []

        scenario_id = self.cur_scenario.get('scenario_id', str(scenario_idx))
        print(f"[EduEnv] reset → scenario {scenario_id} | "
              f"DM={self.controlled_agent_id} | turn_cap={self.turn_cap}")

        # Let agents before the decision_maker take their opening turns
        dm_pos = self.turn_order.index(self.controlled_agent_id)
        for pos in range(dm_pos):
            agent_id = self.turn_order[pos]
            self._other_agent_act(agent_id)

        return self.get_state()

    # ── generation helpers ────────────────────────────────────────────────────

    def _generate_with_prefix(self, agent_id: str, action, prefixes) -> str:
        """Generate the controlled agent's response using prefix embeddings."""
        prompt = self._build_prompt_str(agent_id, settle_allowed=True)
        input_ids, _ = self._get_input_tensors(prompt)
        dev = self.model.base_model.device
        input_ids = input_ids.to(dev)

        with torch.no_grad():
            if not self.test_baseline:
                new_embeddings = self.model.embed_action(action, input_ids).to(
                    dtype=next(self.model.base_model.parameters()).dtype
                )
                output = self.model.base_model.generate(
                    inputs_embeds=new_embeddings,
                    max_new_tokens=300,
                    temperature=self.temperature if self.temperature > 0 else None,
                    do_sample=(self.temperature > 0),
                    pad_token_id=self.model.tokenizer.eos_token_id,
                )
            else:
                output = self.model.base_model.generate(
                    input_ids=input_ids,
                    max_new_tokens=300,
                    temperature=self.temperature if self.temperature > 0 else None,
                    do_sample=(self.temperature > 0),
                    pad_token_id=self.model.tokenizer.eos_token_id,
                )

        raw = self.model.tokenizer.batch_decode(output, skip_special_tokens=True)[0]
        if raw.startswith(prompt):
            raw = raw[len(prompt):]
        return raw.strip()

    def _other_agent_act(self, agent_id: str) -> str:
        """Have a non-controlled agent take its turn without prefix."""
        prompt = self._build_prompt_str(agent_id, settle_allowed=False)
        input_ids, attention_mask = self._get_input_tensors(prompt)
        dev = self.model.base_model.device
        input_ids = input_ids.to(dev)
        attention_mask = attention_mask.to(dev)

        with torch.no_grad():
            output = self.model.base_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=256,
                temperature=self.temperature if self.temperature > 0 else None,
                do_sample=(self.temperature > 0),
                pad_token_id=self.model.tokenizer.eos_token_id,
            )

        raw = self.model.tokenizer.batch_decode(output, skip_special_tokens=True)[0]
        if raw.startswith(prompt):
            raw = raw[len(prompt):]
        raw = raw.strip()

        parsed = _extract_json(raw)
        parsed["speaker"] = agent_id
        parsed["turn"] = self.global_turn
        parsed["raw_output"] = raw
        self.transcript.append(parsed)
        self.global_turn += 1
        return raw

    # ── step ──────────────────────────────────────────────────────────────────

    def step(
        self,
        cur_state,
        action,
        prefixes=None,
        log_time: bool = False,
        save_his: bool = True,
    ):
        """
        Execute one round: decision_maker turn (with prefix) + subsequent
        non-DM turns until it's the DM's turn again (or episode ends).

        Returns: (next_state, reward, done, info)
        """
        if hasattr(cur_state, 'detach'):
            self.cur_states.append(cur_state.detach().cpu().numpy())
        if hasattr(action, 'detach'):
            self.actions.append(action.detach().cpu().numpy())

        t0 = time.time()

        # ── decision_maker takes its turn ──────────────────────────────────────
        raw = self._generate_with_prefix(self.controlled_agent_id, action, prefixes)
        parsed = _extract_json(raw)
        parsed["speaker"] = self.controlled_agent_id
        parsed["turn"] = self.global_turn
        parsed["raw_output"] = raw
        self.transcript.append(parsed)
        self.global_turn += 1

        action_type = parsed.get("type", "say")
        if log_time:
            print(f"  DM ({self.controlled_agent_id}) → type={action_type} "
                  f"[{time.time()-t0:.1f}s]")

        # Terminal: DM settled
        if action_type == "settle":
            reward = _compute_reward(self.cur_scenario, self.transcript)
            done = True
            next_state = self.get_state()
            self._record(next_state, reward, done)
            self._maybe_save(save_his)
            return next_state, reward, done, None

        # Terminal: turn cap hit by DM
        if self.global_turn >= self.turn_cap:
            reward = 0.0
            done = True
            next_state = self.get_state()
            self._record(next_state, reward, done)
            self._maybe_save(save_his)
            return next_state, reward, done, None

        # ── other agents take their turns ──────────────────────────────────────
        dm_pos = self.turn_order.index(self.controlled_agent_id)
        n = len(self.turn_order)
        for offset in range(1, n):
            if self.global_turn >= self.turn_cap:
                break
            other_id = self.turn_order[(dm_pos + offset) % n]
            self._other_agent_act(other_id)

        # ── next state for the DM ──────────────────────────────────────────────
        done = (self.global_turn >= self.turn_cap)
        reward = _compute_reward(self.cur_scenario, self.transcript) if done else 0.0
        next_state = self.get_state()
        self._record(next_state, reward, done)

        if done:
            self._maybe_save(save_his)

        return next_state, reward, done, None

    # ── internal ──────────────────────────────────────────────────────────────

    def _record(self, next_state, reward, done):
        if hasattr(next_state, 'detach'):
            self.next_states.append(next_state.detach().cpu().numpy())
        self.cur_rewards.append(reward)
        self.terminations.append(done)

    def _maybe_save(self, save_his: bool):
        """Save the episode transcript + transitions as a replay buffer .pkl file."""
        if not save_his or not self.saving_dir:
            return
        if not self.cur_states:
            return
        episode = {
            'observations':      np.array(self.cur_states),
            'actions':           np.array(self.actions),
            'rewards':           np.array(self.cur_rewards),
            'next_observations': np.array(self.next_states),
            'terminals':         np.array(self.terminations),
            'transcript':        self.transcript,
            'scenario_id':       self.cur_scenario.get('scenario_id', ''),
        }
        fname = f"ep_{self.cur_scenario.get('scenario_id','?')}_{int(time.time())}.pkl"
        path = os.path.join(self.saving_dir, fname)
        with open(path, 'wb') as f:
            pickle.dump(episode, f)
        final_reward = self.cur_rewards[-1] if self.cur_rewards else 0.0
        print(f"[EduEnv] Saved → {path} | reward={final_reward:.3f} | "
              f"turns={self.global_turn}")

