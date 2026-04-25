import asyncio
from trebek.pipeline.orchestrator import run_pipeline

if __name__ == "__main__":
    asyncio.run(run_pipeline(mode="daemon"))
