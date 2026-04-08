"""
Inference Script — Traffic Signal Control Environment
=====================================================
Runs an LLM agent against three tasks: easy, medium, hard.
Emits mandatory [START] / [STEP] / [END] stdout format.

Required env vars:
    HF_TOKEN       Hugging Face API key
    API_BASE_URL   LLM endpoint  (default: HF router)
    MODEL_NAME     Model identifier (default: Qwen2.5-72B-Instruct)
"""

import asyncio
import json
import os
import textwrap
from typing import List, Optional

from openai import OpenAI
from traffic_env import TrafficAction, TrafficEnv
from traffic_env.models import IntersectionPhaseDecision

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

IMAGE_NAME    = os.getenv("LOCAL_IMAGE_NAME")
API_KEY       = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL  = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME    = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
BENCHMARK     = "traffic_env"

MAX_STEPS             = 50
TEMPERATURE           = 0.2   # low — we want deterministic decisions
MAX_TOKENS            = 300
SUCCESS_SCORE_THRESHOLD = 0.4

# Reward is in [-1, 1]. Normalise episode score to [0, 1].
# Theoretical best per step is ~0.2 (all lanes clear). Over 50 steps = 10.
MAX_TOTAL_REWARD = MAX_STEPS * 0.2

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
    You are a traffic signal controller for a 4-way intersection.
    This is a left-hand traffic network (Indian road model).
    Left turns are always open. You control which other movements get a green light.

    Each step you receive the current network state as JSON.
    You must respond with a JSON object — nothing else, no explanation, no markdown.

    Response format:
    {
        "decisions": [
            {"intersection_id": "<id>", "phase_id": "<phase_id>"}
        ]
    }

    Available phases for the main intersection:
    - "ALL_RED"      : All signals red. Use for transitions.
    - "NS_THROUGH"   : North-South through traffic + left turns.
    - "EW_THROUGH"   : East-West through traffic + left turns.
    - "N_RIGHT"      : North right turn + left turns.
    - "S_RIGHT"      : South right turn + left turns.

    Strategy:
    - Prioritise phases that drain the longest waiting queues.
    - Avoid leaving a direction waiting too long (starvation).
    - In hard task, a surge may hit one road — react quickly.
    - Do not stay on ALL_RED unnecessarily.

    You will be penalised for high waiting queues and rewarded for clearing them.
""").strip()


def build_user_prompt(obs_json: str, step: int, last_reward: float, history: List[str]) -> str:
    history_block = "\n".join(history[-4:]) if history else "None"
    return textwrap.dedent(f"""
        Step: {step}
        Last reward: {last_reward:.2f}
        Recent history:
        {history_block}

        Current network state:
        {obs_json}

        Respond with your phase decision JSON now.
    """).strip()


# ---------------------------------------------------------------------------
# Logging — mandatory format
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    # Sanitise action string — no newlines allowed on a single log line
    action_str = action.replace("\n", " ").replace("\r", "")
    print(
        f"[STEP] step={step} action={action_str} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# LLM call + action parsing
# ---------------------------------------------------------------------------

def get_action(
    client: OpenAI,
    obs_json: str,
    step: int,
    last_reward: float,
    history: List[str],
    intersection_id: str,
) -> tuple[TrafficAction, str, Optional[str]]:
    """
    Ask the LLM for a phase decision.
    Returns (TrafficAction, raw_response_str, error_or_None).
    Falls back to NS_THROUGH on any parse failure.
    """
    user_prompt = build_user_prompt(obs_json, step, last_reward, history)
    raw = ""
    error = None

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        raw = (completion.choices[0].message.content or "").strip()

        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        parsed = json.loads(raw)
        decisions = [
            IntersectionPhaseDecision(
                intersection_id=d["intersection_id"],
                phase_id=d["phase_id"],
            )
            for d in parsed["decisions"]
        ]
        return TrafficAction(decisions=decisions), raw, None

    except Exception as exc:
        error = str(exc)
        print(f"[DEBUG] Parse failed: {exc} | raw: {raw!r}", flush=True)
        # Safe fallback
        fallback = TrafficAction(decisions=[
            IntersectionPhaseDecision(
                intersection_id=intersection_id,
                phase_id="NS_THROUGH",
            )
        ])
        return fallback, raw, error


# ---------------------------------------------------------------------------
# Single task episode
# ---------------------------------------------------------------------------

# async def run_task(env: TrafficEnv, client: OpenAI, task: str) -> None:
#     rewards: List[float] = []
#     history: List[str]   = []
#     steps_taken = 0
#     score = 0.0
#     success = False

#     log_start(task=task, env=BENCHMARK, model=MODEL_NAME)

#     try:
#         result     = await env.reset(task=task)
#         obs        = result.observation
#         last_reward = 0.0

#         print(result)


    #     # Get the main intersection id from the first observation
    #     intersection_id = (
    #         obs.road_network.intersections[0].id
    #         if obs.road_network.intersections
    #         else "intersection_center"
    #     )

    #     for step in range(1, MAX_STEPS + 1):
    #         if result.done:
    #             break

    #         # Serialise only the parts the LLM needs — trim noise
    #         obs_summary = _summarise_obs(obs)
    #         obs_json    = json.dumps(obs_summary, indent=2)

    #         action, raw, error = get_action(
    #             client, obs_json, step, last_reward, history, intersection_id
    #         )

    #         result      = await env.step(action)
    #         obs         = result.observation
    #         reward      = result.reward or 0.0
    #         done        = result.done

    #         rewards.append(reward)
    #         steps_taken = step
    #         last_reward = reward

    #         # Compact action string for log line
    #         action_str = json.dumps(
    #             [{"i": d.intersection_id, "p": d.phase_id} for d in action.decisions]
    #         ).replace(" ", "")

    #         log_step(step=step, action=action_str, reward=reward, done=done, error=error)
    #         history.append(f"Step {step}: phase={action_str} reward={reward:+.2f}")

    #         if done:
    #             break

    #     # Normalise to [0, 1]
    #     # Rewards are in [-1, 1]. Shift to [0, 2] then normalise by 2*MAX_STEPS.
    #     shifted = sum(r + 1.0 for r in rewards)
    #     score   = shifted / (2.0 * MAX_STEPS)
    #     score   = min(max(score, 0.0), 1.0)
    #     success = score >= SUCCESS_SCORE_THRESHOLD

    # except Exception as exc:
    #     print(f"[DEBUG] Episode error: {exc}", flush=True)

    # finally:
    #     log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


def _summarise_obs(obs) -> dict:
    """
    Compact observation for the LLM — just what it needs to decide.
    Full Pydantic object would be too verbose.
    """
    roads_summary = []
    for road in obs.road_network.roads:
        roads_summary.append({
            "id":      road.id,
            "waiting": road.waiting,
            "inflight": road.inflight,
        })

    intersections_summary = []
    for ix in obs.road_network.intersections:
        intersections_summary.append({
            "id":            ix.id,
            "current_phase": ix.phase,
            "valid_phases":  [p.id for p in ix.phase_set],
        })

    return {
        "task":          obs.task,
        "step":          obs.step,
        "roads":         roads_summary,
        "intersections": intersections_summary,
    }


# ---------------------------------------------------------------------------
# Main — run all three tasks sequentially
# ---------------------------------------------------------------------------

async def run_task(env: TrafficEnv, client: OpenAI, task: str) -> None:
    # rewards: List[float] = []
    # history: List[str]   = []
    # steps_taken = 0
    # score = 0.0
    # success = False

    # log_start(task=task, env=BENCHMARK, model=MODEL_NAME)

    try:
        result     = await env.reset(task=task)
        # obs        = result.observation
        last_reward = 0.0
        print("Worked atleast")
        print(result)

    except Exception as exc:
        print(f"[DEBUG] Episode error: {exc}", flush=True)
        pass
    finally:
        # log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
        pass


async def main() -> None:
    base_url = "http://0.0.0.0:8000"
    env = TrafficEnv(base_url=base_url)

    await run_task(env, None, 'easy')
    # async with env:  # connect once
    #     for task in ["easy", "medium", "hard"]:
    #         await run_task(env, client, task)


if __name__ == "__main__":
    asyncio.run(main())