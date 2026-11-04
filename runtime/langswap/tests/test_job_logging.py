from langswap.utils.logging import Logger


def test_job_logs_are_isolated_and_handler_reused(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    first, second = Logger(str(a)), Logger(str(b))
    repeated = Logger(str(a))
    first.file_logger.warning("only job a")
    second.file_logger.warning("only job b")
    assert first.file_logger is repeated.file_logger
    assert len(first.file_logger.handlers) == 1
    assert "only job a" in (a / "logs.log").read_text(encoding="utf-8")
    assert "only job b" not in (a / "logs.log").read_text(encoding="utf-8")
    assert "only job a" not in (b / "logs.log").read_text(encoding="utf-8")
