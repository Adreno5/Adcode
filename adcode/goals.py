"""Durable goals, persisted locally and injected into every system prompt."""

from __future__ import annotations

import json
import os
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

GoalStatus = Literal['active', 'paused', 'completed', 'blocked', 'cancelled']
VALID_STATUSES = {'active', 'paused', 'completed', 'blocked', 'cancelled'}


@dataclass
class Goal:
    id: str
    objective: str
    status: GoalStatus = 'active'
    created_at: str = ''
    updated_at: str = ''
    session: str = ''
    result: str = ''

    @classmethod
    def from_dict(cls, raw: dict) -> 'Goal':
        status = str(raw.get('status', 'active'))
        if status not in VALID_STATUSES:
            status = 'active'
        return cls(
            id=str(raw.get('id') or uuid.uuid4().hex),
            objective=str(raw.get('objective', '')).strip(),
            status=status,  # type: ignore[arg-type]
            created_at=str(raw.get('created_at') or _now()),
            updated_at=str(raw.get('updated_at') or _now()),
            session=str(raw.get('session', '')),
            result=str(raw.get('result', '')),
        )

    def as_dict(self) -> dict:
        return asdict(self)


def _now() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%S%z')


class GoalStore:
    def __init__(self, path: str | os.PathLike[str] = 'sessions/goals.json') -> None:
        self.path = Path(path)
        self.goals: list[Goal] = []
        self.load()

    def load(self) -> list[Goal]:
        try:
            raw = json.loads(self.path.read_text(encoding='utf-8-sig'))
        except (FileNotFoundError, UnicodeError, json.JSONDecodeError, OSError):
            raw = []
        if isinstance(raw, dict):
            raw = raw.get('goals', [])
        if not isinstance(raw, list):
            raw = []
        self.goals = [Goal.from_dict(item) for item in raw if isinstance(item, dict)]
        self.goals = [goal for goal in self.goals if goal.objective]
        return self.goals

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            {'version': 1, 'goals': [goal.as_dict() for goal in self.goals]},
            ensure_ascii=False, indent=2,
        )
        fd, temp_name = tempfile.mkstemp(
            prefix=f'.{self.path.name}.', suffix='.tmp', dir=str(self.path.parent))
        try:
            with os.fdopen(fd, 'w', encoding='utf-8', newline='\n') as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_name, self.path)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)

    def active(self) -> Goal | None:
        for goal in reversed(self.goals):
            if goal.status == 'active':
                return goal
        return None

    def get(self, goal_id: str) -> Goal | None:
        return next((goal for goal in self.goals if goal.id == goal_id), None)

    def create(self, objective: str, session: str = '') -> Goal:
        objective = objective.strip()
        if not objective:
            raise ValueError('goal objective must not be empty')
        current = self.active()
        if current is not None:
            current.status = 'paused'
            current.updated_at = _now()
        now = _now()
        goal = Goal(id=uuid.uuid4().hex[:12], objective=objective,
                    created_at=now, updated_at=now, session=session)
        self.goals.append(goal)
        self.save()
        return goal

    def update(self, goal: Goal, *, status: GoalStatus | None = None,
               result: str | None = None, session: str | None = None) -> Goal:
        if status is not None:
            goal.status = status
        if result is not None:
            goal.result = result
        if session is not None:
            goal.session = session
        goal.updated_at = _now()
        self.save()
        return goal

    def remove(self, goal: Goal) -> None:
        self.goals = [item for item in self.goals if item.id != goal.id]
        self.save()

    @staticmethod
    def prompt(goal: Goal | None) -> str:
        if goal is None or goal.status != 'active':
            return ''
        return (
            '<active_goal>\n'
            f'Goal id: {goal.id}\n'
            f'Objective: {goal.objective}\n'
            'Treat this objective as durable across turns. Keep working until it '
            'is completed or genuinely blocked, then report the concrete result.\n'
            '</active_goal>'
        )
