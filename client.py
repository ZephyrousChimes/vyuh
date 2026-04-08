# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Traffic Env Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import TrafficAction, TrafficObservation, RoadNetwork


class TrafficEnv(
    EnvClient[TrafficAction, TrafficObservation, State]
):
    """Client for the Traffic Signal Control Environment."""

    def _step_payload(self, action: TrafficAction) -> Dict:
        """Convert TrafficAction to JSON payload."""
        return {
            "decisions": [
                {
                    "intersection_id": d.intersection_id,
                    "phase_id": d.phase_id,
                }
                for d in action.decisions
            ]
        }

    def _parse_result(self, payload: Dict) -> StepResult[TrafficObservation]:
        """Parse server response into StepResult[TrafficObservation]."""
        obs_data = payload.get("observation", payload)  # some servers nest, some don't

        observation = TrafficObservation(
            road_network=RoadNetwork(**obs_data["road_network"]),
            task=obs_data.get("task", "easy"),
            step=obs_data.get("step", 0),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """Parse server response into State object."""
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )