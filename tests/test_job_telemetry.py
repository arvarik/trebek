import pytest
from pydantic import ValidationError

def test_job_telemetry_schema_exists() -> None:
    try:
        from trebek.schemas import JobTelemetry
    except ImportError:
        pytest.fail("JobTelemetry schema is missing from trebek.schemas")
        
    # Attempt to instantiate with valid data
    telemetry = JobTelemetry(
        episode_id="ep_123",
        peak_vram_mb=4500.5,
        avg_gpu_utilization_pct=88.2,
        stage_ingestion_ms=120.0,
        gemini_total_input_tokens=1500,
        pydantic_retry_count=1
    )
    assert telemetry.episode_id == "ep_123"

def test_job_telemetry_validation_rules() -> None:
    from trebek.schemas import JobTelemetry
    
    with pytest.raises(ValidationError):
        # Latency cannot be negative
        JobTelemetry(
            episode_id="ep_123",
            stage_ingestion_ms=-50.0 
        )

@pytest.mark.asyncio
async def test_insert_job_telemetry(memory_db_path: str) -> None:
    from trebek.core_database import DatabaseWriter
    from trebek.schemas import JobTelemetry
    
    writer = DatabaseWriter(memory_db_path)
    await writer.start()
    
    try:
        # Requires a pipeline_state entry first due to foreign key constraint
        await writer.execute("INSERT INTO pipeline_state (episode_id, status) VALUES (?, ?)", ("ep_123", "COMPLETED"))
        
        telemetry = JobTelemetry(
            episode_id="ep_123",
            peak_vram_mb=4500.5,
            avg_gpu_utilization_pct=88.2,
            stage_ingestion_ms=120.0,
            gemini_total_input_tokens=1500,
            pydantic_retry_count=1
        )
        
        # Check if the method exists (will fail if not implemented)
        assert hasattr(writer, "insert_job_telemetry"), "DatabaseWriter missing insert_job_telemetry method"
        
        await writer.insert_job_telemetry(telemetry)
        
        # Verify insertion
        result = await writer.execute("SELECT peak_vram_mb, gemini_total_input_tokens FROM job_telemetry WHERE episode_id = ?", ("ep_123",))
        assert result[0][0] == 4500.5
        assert result[0][1] == 1500
    finally:
        await writer.stop()
