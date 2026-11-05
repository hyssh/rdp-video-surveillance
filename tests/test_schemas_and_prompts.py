import importlib
import re

import pytest
import yaml

from conftest import ORCH, ROOT
from postgresql import schemas


def test_default_schema_uses_3072_halfvec_hnsw():
    sql = "\n".join(schemas.build_schema_statements(3072))
    assert "embedding vector(3072)" in sql
    assert "1536" not in sql
    assert "ivfflat" not in sql
    assert "hnsw ((embedding::halfvec(3072)) halfvec_l2_ops)" in sql


def test_small_dims_index_vector_directly():
    sql = "\n".join(schemas.build_schema_statements(1536))
    assert "embedding vector(1536)" in sql
    assert "hnsw (embedding vector_l2_ops)" in sql
    assert "halfvec" not in sql


def test_oversized_dims_have_no_index():
    sql = "\n".join(schemas.build_schema_statements(5096))
    assert "USING hnsw" not in sql
    assert "embedding vector(5096)" in sql


def test_migration_resets_column_when_dimension_changes():
    sql = "\n".join(schemas.build_schema_statements(3072))
    assert "DROP INDEX IF EXISTS idx_frames_embedding" in sql
    assert "ALTER COLUMN embedding TYPE vector(3072) USING NULL" in sql


def test_search_function_casts_to_halfvec_for_large_dims():
    sql = schemas.build_search_function_sql(3072)
    assert "query_vector vector," in sql
    assert "f.embedding::halfvec(3072)" in sql
    assert "1536" not in sql


def test_search_function_plain_for_small_dims():
    assert "halfvec" not in schemas.build_search_function_sql(1536)


def test_no_hardcoded_1536_in_orchestrator_code():
    for path in ORCH.rglob("*.py"):
        assert "vector(1536)" not in path.read_text(encoding="utf-8"), path


@pytest.mark.parametrize("module", [
    "agents.chat_completion.chat_completion_agent",
    "agents.security_reviewer.security_reviewer_agent",
    "agents.image_analyzer.image_analyzer_agent",
])
def test_prompts_load_independent_of_cwd(tmp_path, monkeypatch, module):
    mod = importlib.import_module(module)
    monkeypatch.chdir(tmp_path)
    assert len(mod.load_prompts()) > 100


@pytest.mark.parametrize("name", ["chat_completion", "image_analyzer", "security_reviewer"])
def test_prompts_have_no_placeholders_or_typos(name):
    text = (ORCH / "agents" / name / "agent.yml").read_text(encoding="utf-8")
    assert yaml.safe_load(text)
    assert "descriotion" not in text
    assert '"fame"' not in text
    assert not re.search(r"^\s*\|\s*\.\.\.", text, re.M), "placeholder rubric row"
    if name != "security_reviewer":
        for level in range(0, 11):
            assert re.search(rf"^\s*\|\s*{level}\s+\|", text, re.M), f"rubric level {level} missing"


def _env_names_used_by_code():
    code = "".join(p.read_text(encoding="utf-8") for p in ORCH.rglob("*.py"))
    used = set(re.findall(r"getenv\(\s*[\"']([A-Z0-9_]+)[\"']", code))
    for call in re.findall(r"env_first\(([^)]*)\)", code):
        used |= set(re.findall(r"\"([A-Z][A-Z0-9_]+)\"", call))
    return used


def test_env_vars_in_code_are_documented_and_passed_to_compose():
    used = _env_names_used_by_code()
    example = set(re.findall(r"^([A-Z0-9_]+)=", (ROOT / ".env.exmaple").read_text(encoding="utf-8"), re.M))
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    block = compose.split("  orchestrator:")[1].split("healthcheck")[0]
    passed = set(re.findall(r"-\s*([A-Z0-9_]+)=", block))
    optional = {"LOG_LEVEL", "UVICORN_RELOAD"}
    assert not used - example - optional, sorted(used - example - optional)
    assert not used - passed - optional, sorted(used - passed - optional)


def test_requirements_cover_runtime_imports():
    reqs = (ORCH / "requirements.txt").read_text(encoding="utf-8").lower()
    for pkg in ("azure-identity", "azure-storage-blob", "semantic-kernel", "psycopg2", "opencv", "pyyaml", "pymeta3"):
        assert pkg in reqs, pkg
