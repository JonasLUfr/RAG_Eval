from pathlib import Path
from uuid import uuid4

import pandas as pd

from app.core.config import AppConfig
from app.models import EvalResult, EvalSample, ExperimentRun, ProjectContext, ScoreItem, SystemResponse
from app.services.importer import dataframe_to_responses
from app.services.testset_generator import TestsetGenerator as SampleGenerator
from app.storage import SQLiteStore


def test_deduplicate_keeps_first_question():
    samples = [
        EvalSample(question="订单 状态 是什么？", reference_answer="A"),
        EvalSample(question="订单状态是什么？", reference_answer="B"),
    ]

    unique = SampleGenerator(AppConfig()).deduplicate(samples)

    assert len(unique) == 1
    assert unique[0].reference_answer == "A"


def test_dataframe_to_responses_parses_lists_and_numbers():
    df = pd.DataFrame(
        [
            {
                "question_id": "q1",
                "question": "问题",
                "reference_answer": "参考",
                "answer": "回答",
                "retrieved_contexts": '["ctx1", "ctx2"]',
                "citations": "cite1\ncite2",
                "latency_ms": "12.5",
                "token_usage": "42",
            }
        ]
    )

    responses = dataframe_to_responses(df)

    assert len(responses) == 1
    assert responses[0].retrieved_contexts == ["ctx1", "ctx2"]
    assert responses[0].citations == ["cite1", "cite2"]
    assert responses[0].latency_ms == 12.5
    assert responses[0].token_usage == 42


def test_sqlite_store_roundtrip():
    runtime_dir = Path("tests/.runtime")
    runtime_dir.mkdir(parents=True, exist_ok=True)
    db_path = runtime_dir / f"rag_eval_{uuid4().hex}.sqlite3"
    store = SQLiteStore(db_path)
    project = ProjectContext(name="测试项目")
    sample = EvalSample(question_id="q1", question="问题", review_status="已通过")
    run = ExperimentRun(run_id="run1", project_id=project.project_id, name="实验")
    response = SystemResponse(response_id="resp1", question_id="q1", question="问题", answer="回答")
    result = EvalResult(
        result_id="eval1",
        question_id="q1",
        response_id="resp1",
        scores={"correctness": ScoreItem(raw_score=1, normalized_score=1, reason="ok")},
        normalized_score=1.0,
        evaluation_status="scored",
    )

    store.save_project_context(project)
    store.save_test_cases(project.project_id, [sample])
    store.save_experiment(run)
    store.save_system_responses(run.run_id, [response])
    store.save_eval_results(run.run_id, [result])

    assert store.get_project_context(project.project_id).name == "测试项目"
    assert store.list_test_cases(project.project_id, approved_only=True)[0].question == "问题"
    assert store.get_experiment("run1").name == "实验"
    assert store.list_system_responses("run1")[0].answer == "回答"
    loaded_result = store.list_eval_results("run1")[0]
    assert loaded_result.scores["correctness"].normalized_score == 1
    assert loaded_result.evaluation_status == "scored"


def test_sqlite_store_delete_project_removes_related_records():
    runtime_dir = Path("tests/.runtime")
    runtime_dir.mkdir(parents=True, exist_ok=True)
    db_path = runtime_dir / f"rag_eval_{uuid4().hex}.sqlite3"
    store = SQLiteStore(db_path)
    project = ProjectContext(name="Project to delete")
    sample = EvalSample(question_id="q1", question="Question", review_status="approved")
    run = ExperimentRun(run_id="run-delete", project_id=project.project_id, name="Run")
    response = SystemResponse(response_id="resp-delete", question_id="q1", question="Question", answer="Answer")
    result = EvalResult(
        result_id="eval-delete",
        question_id="q1",
        response_id="resp-delete",
        scores={"correctness": ScoreItem(raw_score=1, normalized_score=1, reason="ok")},
        normalized_score=1.0,
        evaluation_status="scored",
    )

    store.save_project_context(project)
    store.save_test_cases(project.project_id, [sample])
    store.save_experiment(run)
    store.save_system_responses(run.run_id, [response])
    store.save_eval_results(run.run_id, [result])

    store.delete_project(project.project_id)

    assert store.get_project_context(project.project_id) is None
    assert store.list_test_cases(project.project_id) == []
    assert store.list_experiments(project.project_id) == []
    assert store.get_experiment(run.run_id) is None
    assert store.list_system_responses(run.run_id) == []
    assert store.list_eval_results(run.run_id) == []
