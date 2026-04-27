PRAGMA auto_vacuum = INCREMENTAL;
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;

CREATE TABLE IF NOT EXISTS pipeline_state (
    episode_id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    source_filename TEXT,
    transcript_path TEXT,
    retry_count INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS episodes (
    episode_id TEXT PRIMARY KEY,
    air_date DATE,
    host_name TEXT,
    is_tournament BOOLEAN
);

CREATE TABLE IF NOT EXISTS contestants (
    contestant_id TEXT PRIMARY KEY,
    name TEXT,
    occupational_category TEXT,
    is_returning_champion BOOLEAN
);

CREATE TABLE IF NOT EXISTS episode_performances (
    episode_id TEXT REFERENCES episodes(episode_id),
    contestant_id TEXT REFERENCES contestants(contestant_id),
    podium_position INTEGER CHECK(podium_position IN (1, 2, 3)),
    coryat_score INTEGER,
    final_score INTEGER,
    forrest_bounce_index REAL,
    PRIMARY KEY (episode_id, contestant_id)
);

CREATE TABLE IF NOT EXISTS clues (
    clue_id TEXT PRIMARY KEY,
    episode_id TEXT REFERENCES episodes(episode_id),
    round TEXT CHECK(round IN ('Jeopardy', 'Double Jeopardy', 'Final Jeopardy', 'Tiebreaker')),
    category TEXT,
    board_row INTEGER,
    board_col INTEGER,
    selection_order INTEGER,
    
    clue_text TEXT,
    correct_response TEXT,
    is_daily_double BOOLEAN,
    is_triple_stumper BOOLEAN,
    daily_double_wager TEXT,
    wagerer_name TEXT,
    
    requires_visual_context BOOLEAN,
    host_start_timestamp_ms REAL,
    host_finish_timestamp_ms REAL,
    clue_syllable_count INTEGER,
    host_speech_rate_wpm REAL,
    
    selector_had_board_control BOOLEAN,

    clue_embedding BLOB,
    response_embedding BLOB,
    semantic_lateral_distance REAL
);

CREATE TABLE IF NOT EXISTS buzz_attempts (
    attempt_id TEXT PRIMARY KEY,
    clue_id TEXT REFERENCES clues(clue_id),
    contestant_id TEXT REFERENCES contestants(contestant_id),
    
    attempt_order INTEGER,
    buzz_timestamp_ms REAL,
    podium_light_timestamp_ms REAL,
    true_buzzer_latency_ms REAL,
    is_lockout_inferred BOOLEAN,
    
    response_given TEXT,
    is_correct BOOLEAN,
    response_start_timestamp_ms REAL, 
    brain_freeze_duration_ms REAL,
    true_acoustic_confidence_score REAL,
    disfluency_count INTEGER,
    phonetic_similarity_score REAL
);

CREATE TABLE IF NOT EXISTS wagers (
    wager_id TEXT PRIMARY KEY,
    clue_id TEXT REFERENCES clues(clue_id),
    contestant_id TEXT REFERENCES contestants(contestant_id),
    
    running_score_at_time INTEGER,
    opponent_1_score INTEGER,
    opponent_2_score INTEGER,
    actual_wager INTEGER,
    
    game_theory_optimal_wager INTEGER,
    wager_irrationality_delta INTEGER
);

CREATE TABLE IF NOT EXISTS score_adjustments (
    adjustment_id TEXT PRIMARY KEY,
    episode_id TEXT REFERENCES episodes(episode_id),
    contestant_id TEXT REFERENCES contestants(contestant_id),
    points_adjusted INTEGER,
    reason TEXT,
    effective_after_clue_selection_order INTEGER
);

CREATE TABLE IF NOT EXISTS job_telemetry (
    episode_id TEXT PRIMARY KEY REFERENCES pipeline_state(episode_id),
    
    -- Hardware Resource Signatures
    peak_vram_mb REAL,
    avg_gpu_utilization_pct REAL,

    -- Pipeline Stage Execution Latency (ms)
    stage_ingestion_ms REAL,
    stage_gpu_extraction_ms REAL,
    stage_commercial_filtering_ms REAL,
    stage_structured_extraction_ms REAL,
    stage_multimodal_ms REAL,
    stage_vectorization_ms REAL,

    -- Gemini AI Telemetry & Self-Healing
    gemini_total_input_tokens INTEGER,
    gemini_total_output_tokens INTEGER,
    gemini_total_cached_tokens INTEGER,
    gemini_total_cost_usd REAL,
    gemini_api_latency_ms REAL,
    pydantic_retry_count INTEGER,
    
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Performance indexes for pipeline polling and analytical queries
CREATE INDEX IF NOT EXISTS idx_pipeline_state_status ON pipeline_state(status, created_at);
CREATE INDEX IF NOT EXISTS idx_clues_episode_id ON clues(episode_id);
CREATE INDEX IF NOT EXISTS idx_buzz_attempts_clue_id ON buzz_attempts(clue_id);
CREATE INDEX IF NOT EXISTS idx_buzz_attempts_contestant_id ON buzz_attempts(contestant_id);
CREATE INDEX IF NOT EXISTS idx_wagers_clue_id ON wagers(clue_id);
CREATE INDEX IF NOT EXISTS idx_score_adjustments_episode_id ON score_adjustments(episode_id);
CREATE INDEX IF NOT EXISTS idx_job_telemetry_episode_id ON job_telemetry(episode_id);