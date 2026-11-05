import importlib
import re
from unittest import mock

import pytest
import yaml

from conftest import ORCH, ROOT
from postgresql import schemas


def _statements(dims):
    """Run initialize_schemas against a fake cursor and return the executed SQL."""
    executed = []

    class FakeDM:
        def __enter__(self):
            return mock.Mock(execute=lambda sql, *a: executed.append(sql))

        def __exit__(self, *exc):
            return False

    with mock.patch("postgresql.connection.DatabaseManager", FakeDM), \
            mock.patch.object(schemas, "get_embedding_dimensions", return_value=dims):
        assert schemas.initialize_schemas()
    return "\n".join(executed)


def test_default_schema_uses_3072_halfvec_hnsw():
    sql = _statements(3072)
    assert "embedding vector(3072)" in sql
    assert "1536" not in sql
    assert "ivfflat" not in sql
    assert "hnsw ((embedding::halfvec(3072)) halfvec_l2_ops)" in sql


def test_small_dims_keep_legacy_ivfflat_index():
    sql = _statements(1536)
    assert "embedding vector(1536)" in sql
    assert "idx_frames_embedding ON frames USING ivfflat (embedding vector_l2_ops)" in sql
    assert "USING hnsw" not in sql


def test_oversized_dims_have_no_index():
    sql = _statements(5096)
    assert "USING hnsw" not in sql and "ivfflat" not in sql
    assert "embedding vector(5096)" in sql


def test_migration_resets_column_when_dimension_changes():
    sql = _statements(3072)
    assert "DROP INDEX IF EXISTS idx_frames_embedding;" in sql
    assert "ALTER TABLE frames ALTER COLUMN embedding TYPE vector(3072) USING NULL" in sql


def test_search_function_casts_to_halfvec_for_large_dims():
    sql = schemas.build_search_function_sql(3072)
    assert "query_vector vector," in sql
    assert "f.embedding::halfvec(3072)" in sql
    assert "1536" not in sql


def test_search_function_plain_for_small_dims():
    assert "halfvec" not in schemas.build_search_function_sql(1536)


@pytest.mark.parametrize("dims,fragment", [(3072, "embedding::halfvec(3072)"), (1536, "(embedding) <->")])
def test_repository_search_matches_index_expression(dims, fragment):
    from postgresql import frame_repository

    captured = []

    class FakeDM:
        def __enter__(self):
            cursor = mock.Mock()
            cursor.execute = lambda sql, params=None: captured.append(sql)
            cursor.fetchall.return_value = []
            return cursor

        def __exit__(self, *exc):
            return False

    with mock.patch("postgresql.connection.DatabaseManager", FakeDM), \
            mock.patch.object(schemas, "get_embedding_dimensions", return_value=dims):
        frame_repository.search_similar_frames([0.0] * dims, None, 3, video_id=1, raise_on_error=True)
    assert fragment in captured[0]


def test_no_hardcoded_1536_in_orchestrator_code():
    for path in ORCH.rglob("*.py"):
        assert not re.search(r"vector\(1536\)\s*[,\n]", path.read_text(encoding="utf-8")), path


@pytest.mark.parametrize("module", [
    "agents.chat_completion.chat_completion_agent",
    "agents.security_reviewer.security_reviewer_agent",
])
def test_prompts_load_independent_of_cwd(tmp_path, monkeypatch, module):
    mod = importlib.import_module(module)
    monkeypatch.chdir(tmp_path)
    assert len(mod.load_prompts()) > 100


@pytest.mark.parametrize("name", ["chat_completion", "image_analyzer", "security_reviewer"])
def test_prompts_parse_and_have_no_typos(name):
    text = (ORCH / "agents" / name / "agent.yml").read_text(encoding="utf-8")
    assert yaml.safe_load(text)
    assert "descriotion" not in text
    assert '"fame"' not in text


def _env_names_used_by_code():
    code = "".join(p.read_text(encoding="utf-8") for p in ORCH.rglob("*.py") if "ui" not in p.parts)
    used = set(re.findall(r"getenv\(\s*[\"']([A-Z0-9_]+)[\"']", code))
    for call in re.findall(r"(?:env_first|_first_env)\(([^)]*)\)", code):
        used |= set(re.findall(r"\"([A-Z][A-Z0-9_]+)\"", call))
    return used


def test_env_vars_in_code_are_documented_and_passed_to_compose():
    used = _env_names_used_by_code()
    example_text = (ROOT / ".env.example").read_text(encoding="utf-8")
    documented = set(re.findall(r"^#?\s*([A-Z][A-Z0-9_]+)=", example_text, re.M))
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    block = compose.split("  orchestrator:")[1].split("healthcheck")[0]
    passed = set(re.findall(r"-\s*([A-Z][A-Z0-9_]+)=", block))
    # legacy aliases are accepted by the code but not required in the example/compose
    aliases = {"AZURE_OPENAI_APIKEY", "AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", "AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME",
               "AZURE_AI_AGENT_CHAT_MODEL_DEPLOYMENT_NAME", "AZURE_AI_AGENT_EMBEDDING_MODEL_DEPLOYMENT_NAME",
               "AZURE_OPENAI_USE_ENTRA", "AZURE_TEXT_EMBEDDING_API_VERSION", "AZURE_TEXT_EMBEDDING_ENDPOINT", "AZURE_TEXT_EMBEDDING_APIKEY", "LOG_LEVEL", "UVICORN_RELOAD", "CLIP_CACHE_DIR"}
    for name in used - aliases:
        if not name.startswith(("AZURE_", "EMBEDDING_", "OMNIPARSER_")):
            continue
        assert name in documented, f"{name} missing from .env.example"
    required_in_compose = {"AZURE_STORAGE_CONNECTION_STRING", "AZURE_STORAGE_ACCOUNT_NAME", "EMBEDDING_DIMENSIONS",
                           "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_CHAT_COMPLETION_MODEL",
                           "AZURE_OPENAI_EMBEDDING_MODEL", "AZURE_CLIENT_ID", "AZURE_TENANT_ID", "AZURE_CLIENT_SECRET"}
    assert not required_in_compose - passed, sorted(required_in_compose - passed)


def test_defaults_are_3072_and_new_models():
    for name in (".env.example", ".env.docker.example"):
        text = (ROOT / name).read_text(encoding="utf-8")
        assert "text-embedding-3-large" in text and "gpt-6.1-sol" in text
        assert re.search(r"^EMBEDDING_DIMENSIONS=3072", text, re.M)
    assert not (ROOT / ".env.exmaple").exists()


def test_requirements_cover_runtime_imports():
    reqs = (ORCH / "requirements.txt").read_text(encoding="utf-8").lower()
    for pkg in ("azure-identity", "azure-storage-blob", "agent-framework", "psycopg2", "opencv", "pyyaml", "pymeta3"):
        assert pkg in reqs, pkg
