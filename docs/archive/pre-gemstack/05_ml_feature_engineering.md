# ML Feature Engineering Pipelines

To train predictive ML models, the raw text provides only 20% of the value. The remaining 80% lies in the physics of the game: reaction times, acoustic hesitation, board-hunting strategies, and mathematical anomalies under pressure.

Here is the breakdown of the cutting-edge feature pipelines designed to make this the most powerful game-theory dataset ever created.

## 1. The Forrest Bounce Index (FBI) Pipeline
* **Concept:** By analyzing the `board_row` against the `selection_order`, we can mathematically map a player's aggression. The State Machine explicitly tracks `current_board_control_contestant_id` (updating only on correct responses) to correctly isolate forced vs. voluntary Forrest Bouncing.
* **Implementation:** A post-processing stage calculates the entropy and physical "jump distance" between consecutive selections made *while holding board control*. This is aggregated into the `forrest_bounce_index` metric in the `episode_performances` table.
* **ML Application:** Train ML models to predict win-rates based purely on board-selection entropy. It isolates the strategy of hunting Daily Doubles from actual trivia knowledge.

## 2. Coryat Score Isolation Pipeline
* **Concept:** The "Coryat Score" is a player's score if wagering was removed entirely. It ignores the luck of finding Daily Doubles or Final Jeopardy wagers, isolating pure trivia knowledge.
* **Implementation:** By strictly tracking every transaction (correct responses minus incorrect responses, ignoring Daily Double wagers), we isolate the Coryat Score directly within our `episode_performances` table.
* **ML Application:** This is the ultimate target variable for predicting a champion's longevity and consistency across multiple episodes, eliminating the noise of game theory luck.

## 3. Semantic Difficulty Index (Split Embeddings)
* **Concept:** A standard database limits you to keyword matches, but Jeopardy clues are highly semantic. Some clues require direct factual recall (low lateral distance), while others rely on heavy wordplay/puns (high lateral distance).
* **Implementation:** We generate two separate embeddings (`clue_embedding` and `response_embedding`) and calculate the `semantic_lateral_distance` (Cosine Distance) between them.
* **ML Application:** We can mathematically prove how "lateral" a question is. Low distance = direct factual recall. High distance = heavy wordplay/puns. This allows us to train models that predict a contestant's likelihood to miss a clue based on its semantic structure, not just its category.

## 4. Game-Theory Wager Irrationality
* **Concept:** When a contestant hits a Daily Double or Final Jeopardy, they must make a wager based on the current mathematical state of the game. LLMs often hallucinate true running scores, corrupting game-theory data.
* **Implementation:** The Deterministic Game-State Verification Engine (Python State Machine) replays extracted atomic events to track exact running scores. It passes this exact score state into a Minimax/Nash Equilibrium algorithm to calculate the `game_theory_optimal_wager`.
* **ML Application:** By subtracting the optimal wager from the actual wager, we calculate the `wager_irrationality_delta`. This isolated metric quantifies human risk aversion, calculation errors, and irrationality under televised pressure.

## 5. True Visual Buzzer Latency (The Lockout Physics)
* **Concept:** Eliminating human variance from reaction times is critical. The host's reading speed fluctuates, making purely audio-based latency calculations inaccurate.
* **Implementation:** Gemini Pro Vision analyzes video frames immediately following the `host_finish_timestamp` to detect the exact frame the "podium indicator lights" illuminate. Latency is calculated as `buzz_timestamp - podium_light_timestamp_ms`.
* **ML Application:** Establishes the exact spatiotemporal reflex time of contestants, eliminating game-show hardware and host-variance noise.

## 6. Deterministic Acoustic Post-Processing
* **Concept:** LLMs are prone to hallucinating counts of "ums" or "uhs". We require deterministic confidence metrics.
* **Implementation:** We inject an `initial_prompt` into WhisperX to prevent it from normalizing out hesitations. A Python utility cross-references the LLM's semantic `BuzzAttempt` boundaries with the raw WhisperX word-level `.prob` logprobs to calculate mathematically true acoustic confidence and deterministic disfluency counts.
* **ML Application:** Provides objective measurements of psychological pressure and brain-freeze durations, devoid of LLM interpretation bias.
