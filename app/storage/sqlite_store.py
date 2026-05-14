from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable

from app.models import EvalResult, EvalSample, ExperimentRun, ProjectContext, SystemResponse
from app.models.schemas import now_iso


class SQLiteStore:
    """轻量 SQLite 存储层；MVP 阶段用 JSON 列保留 schema 演进空间。"""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS project_contexts (
                    project_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    data_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS test_cases (
                    question_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    question TEXT NOT NULL,
                    review_status TEXT NOT NULL,
                    data_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS experiments (
                    run_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    aggregate_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS system_responses (
                    response_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    question_id TEXT NOT NULL,
                    question TEXT NOT NULL,
                    success INTEGER NOT NULL,
                    error TEXT NOT NULL,
                    data_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS eval_results (
                    result_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    question_id TEXT NOT NULL,
                    response_id TEXT NOT NULL,
                    normalized_score REAL NOT NULL,
                    failure_labels_json TEXT NOT NULL,
                    data_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )

    @staticmethod
    def _json(data: dict) -> str:
        return json.dumps(data, ensure_ascii=False)

    @staticmethod
    def _loads(text: str) -> dict:
        return json.loads(text)

    def save_project_context(self, context: ProjectContext) -> None:
        context.updated_at = now_iso()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO project_contexts(project_id, name, data_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(project_id) DO UPDATE SET
                    name=excluded.name,
                    data_json=excluded.data_json,
                    updated_at=excluded.updated_at
                """,
                (
                    context.project_id,
                    context.name,
                    self._json(context.to_dict()),
                    context.created_at,
                    context.updated_at,
                ),
            )

    def list_project_contexts(self) -> list[ProjectContext]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT data_json FROM project_contexts ORDER BY updated_at DESC"
            ).fetchall()
        return [ProjectContext.from_dict(self._loads(row["data_json"])) for row in rows]

    def get_project_context(self, project_id: str) -> ProjectContext | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT data_json FROM project_contexts WHERE project_id=?", (project_id,)
            ).fetchone()
        return ProjectContext.from_dict(self._loads(row["data_json"])) if row else None

    def save_test_cases(self, project_id: str, samples: Iterable[EvalSample]) -> None:
        with self.connect() as conn:
            for sample in samples:
                sample.updated_at = now_iso()
                conn.execute(
                    """
                    INSERT INTO test_cases(question_id, project_id, question, review_status, data_json, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(question_id) DO UPDATE SET
                        question=excluded.question,
                        review_status=excluded.review_status,
                        data_json=excluded.data_json,
                        updated_at=excluded.updated_at
                    """,
                    (
                        sample.question_id,
                        project_id,
                        sample.question,
                        sample.review_status,
                        self._json(sample.to_dict()),
                        sample.updated_at,
                    ),
                )

    def list_test_cases(self, project_id: str, approved_only: bool = False) -> list[EvalSample]:
        sql = "SELECT data_json FROM test_cases WHERE project_id=?"
        params: tuple = (project_id,)
        if approved_only:
            sql += " AND review_status='已通过'"
        sql += " ORDER BY updated_at DESC"
        with self.connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [EvalSample.from_dict(self._loads(row["data_json"])) for row in rows]

    def delete_test_case(self, question_id: str) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM test_cases WHERE question_id=?", (question_id,))

    def save_experiment(self, run: ExperimentRun) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO experiments(run_id, project_id, name, mode, config_json, aggregate_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    name=excluded.name,
                    config_json=excluded.config_json,
                    aggregate_json=excluded.aggregate_json
                """,
                (
                    run.run_id,
                    run.project_id,
                    run.name,
                    run.mode,
                    self._json(run.config),
                    self._json(run.aggregate),
                    run.created_at,
                ),
            )

    def list_experiments(self, project_id: str | None = None) -> list[ExperimentRun]:
        sql = "SELECT * FROM experiments"
        params: tuple = ()
        if project_id:
            sql += " WHERE project_id=?"
            params = (project_id,)
        sql += " ORDER BY created_at DESC"
        with self.connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [
            ExperimentRun(
                run_id=row["run_id"],
                project_id=row["project_id"],
                name=row["name"],
                mode=row["mode"],
                config=self._loads(row["config_json"]),
                aggregate=self._loads(row["aggregate_json"]),
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def get_experiment(self, run_id: str) -> ExperimentRun | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM experiments WHERE run_id=?", (run_id,)).fetchone()
        if not row:
            return None
        return ExperimentRun(
            run_id=row["run_id"],
            project_id=row["project_id"],
            name=row["name"],
            mode=row["mode"],
            config=self._loads(row["config_json"]),
            aggregate=self._loads(row["aggregate_json"]),
            created_at=row["created_at"],
        )

    def save_system_responses(self, run_id: str, responses: Iterable[SystemResponse]) -> None:
        with self.connect() as conn:
            for response in responses:
                conn.execute(
                    """
                    INSERT INTO system_responses(
                        response_id, run_id, question_id, question, success, error, data_json, created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(response_id) DO UPDATE SET
                        success=excluded.success,
                        error=excluded.error,
                        data_json=excluded.data_json
                    """,
                    (
                        response.response_id,
                        run_id,
                        response.question_id,
                        response.question,
                        1 if response.success else 0,
                        response.error,
                        self._json(response.to_dict()),
                        response.created_at,
                    ),
                )

    def list_system_responses(self, run_id: str) -> list[SystemResponse]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT data_json FROM system_responses WHERE run_id=? ORDER BY created_at ASC",
                (run_id,),
            ).fetchall()
        return [SystemResponse.from_dict(self._loads(row["data_json"])) for row in rows]

    def save_eval_results(self, run_id: str, results: Iterable[EvalResult]) -> None:
        with self.connect() as conn:
            for result in results:
                conn.execute(
                    """
                    INSERT INTO eval_results(
                        result_id, run_id, question_id, response_id, normalized_score,
                        failure_labels_json, data_json, created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(result_id) DO UPDATE SET
                        normalized_score=excluded.normalized_score,
                        failure_labels_json=excluded.failure_labels_json,
                        data_json=excluded.data_json
                    """,
                    (
                        result.result_id,
                        run_id,
                        result.question_id,
                        result.response_id,
                        result.normalized_score,
                        self._json(result.failure_labels),
                        self._json(result.to_dict()),
                        result.created_at,
                    ),
                )

    def list_eval_results(self, run_id: str) -> list[EvalResult]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT data_json FROM eval_results WHERE run_id=? ORDER BY created_at ASC",
                (run_id,),
            ).fetchall()
        return [EvalResult.from_dict(self._loads(row["data_json"])) for row in rows]

    def delete_eval_results(self, run_id: str) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM eval_results WHERE run_id=?", (run_id,))

    def delete_experiment(self, run_id: str) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM eval_results WHERE run_id=?", (run_id,))
            conn.execute("DELETE FROM system_responses WHERE run_id=?", (run_id,))
            conn.execute("DELETE FROM experiments WHERE run_id=?", (run_id,))

    def delete_project(self, project_id: str) -> None:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT run_id FROM experiments WHERE project_id=?",
                (project_id,),
            ).fetchall()
            for row in rows:
                run_id = row["run_id"]
                conn.execute("DELETE FROM eval_results WHERE run_id=?", (run_id,))
                conn.execute("DELETE FROM system_responses WHERE run_id=?", (run_id,))
            conn.execute("DELETE FROM experiments WHERE project_id=?", (project_id,))
            conn.execute("DELETE FROM test_cases WHERE project_id=?", (project_id,))
            conn.execute("DELETE FROM project_contexts WHERE project_id=?", (project_id,))

    def update_eval_failure_labels(self, result_id: str, labels: list[str]) -> None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT data_json FROM eval_results WHERE result_id=?", (result_id,)
            ).fetchone()
            if not row:
                return
            data = self._loads(row["data_json"])
            data["failure_labels"] = labels
            conn.execute(
                """
                UPDATE eval_results
                SET failure_labels_json=?, data_json=?
                WHERE result_id=?
                """,
                (self._json(labels), self._json(data), result_id),
            )
