# Data Dictionary and Game Metadata

Data version: `freeze_20261101`. All review creation and comparison dates use UTC.

## Dataset Rules

- `playtime_at_review` and `playtime_forever` are stored in minutes in the raw review data. Hour fields divide by 60.
- `language` is the Steam API language filter (`english` or `schinese`), not a separate language-classification result.
- `partial` means at least one requested language returned fewer than 200 reviews in the frozen 183-day window. It does not by itself mean that an endpoint failed, and no rows are imputed or duplicated.
- `steam_genres` and `steam_categories` come from the frozen Steam Store `appdetails` response. `project_genres` are curated labels in the project configuration and may be more specific, such as FPS or ARPG.
- Use `steam_genres` for official store taxonomy. Use `project_genres` only when the analysis explicitly calls for the curated project grouping.

## Partial Samples

| Game | Partial languages | Counts and reason |
| --- | --- | --- |
| Path of Exile | schinese | schinese: 22/200 reviews; source_total_matching=22 |
| Sid Meier's Civilization V | schinese | schinese: 79/200 reviews; source_total_matching=79 |
| Space Engineers | schinese | schinese: 35/200 reviews; source_total_matching=35 |
| Brawlhalla | schinese | schinese: 150/200 reviews; source_total_matching=150 |

## Path of Exile Store Endpoint

Path of Exile has `store_endpoint_status=ok` and `store_fallback_cc=us`. The initial Store `appdetails` request with `cc=cn` returned an unsuccessful application record, so the collector retried with `cc=us`. The US response succeeded and supplies the genres and other store metadata in `game_meta.csv`. This fallback is unrelated to the review endpoint; the Path of Exile sample is partial because only 22 Simplified Chinese reviews matched the frozen review window.

## Field Dictionary

### review_datasets

| Field | Type | Unit | Source | Transformation | Description | Caveat |
| --- | --- | --- | --- | --- | --- | --- |
| game | string | none | freeze_manifest selected_runs | none | Configured game name. |  |
| appid | integer | none | Steam review endpoint, frozen language reviews.jsonl | none | Steam application identifier. |  |
| language | category | none | Steam review endpoint, frozen language reviews.jsonl | none | Requested Steam review language. | API language filter, not an independently detected language. |
| recommendationid | string | none | Steam review endpoint, frozen language reviews.jsonl | none | Steam review/recommendation identifier. | Kept as text; globally unique in clean_master. |
| steamid | string | none | Steam review endpoint, frozen language reviews.jsonl | none | Steam author identifier. | Identifier, not a numeric measure; do not convert to floating point. |
| review | string | none | Steam review endpoint, frozen language reviews.jsonl | none | Original review text returned by Steam. |  |
| review_clean | string | none | scripts/clean_reviews.py | Collapse all whitespace runs to one space and trim. | Whitespace-normalized review text with original case retained. |  |
| voted_up | boolean | none | Steam review endpoint, frozen language reviews.jsonl | none | Steam recommendation label. | A recommendation indicator, not a continuous sentiment score. |
| timestamp_created | integer | seconds since 1970-01-01 00:00:00 UTC | Steam review endpoint, frozen language reviews.jsonl | none | Review creation time as Unix timestamp. |  |
| timestamp_updated | integer | seconds since 1970-01-01 00:00:00 UTC | Steam review endpoint, frozen language reviews.jsonl | none | Review last-update time as Unix timestamp. |  |
| created_utc | datetime | UTC | scripts/steam_data.py | Convert timestamp_created from Unix seconds to ISO-8601. | ISO-8601 UTC creation timestamp produced during collection. |  |
| created_utc_clean | datetime | UTC | scripts/clean_reviews.py | Reparse timestamp_created and serialize with +00:00 offset. | Validated ISO-8601 creation timestamp. |  |
| created_date_utc | date | UTC calendar date | scripts/clean_reviews.py | Date part of created_utc_clean. | UTC calendar date of review creation. |  |
| timestamp_dev_responded | integer | seconds since 1970-01-01 00:00:00 UTC | Steam review endpoint, frozen language reviews.jsonl | none | Developer response time when provided by Steam. |  |
| developer_response | string | none | Steam review endpoint, frozen language reviews.jsonl | none | Developer response text when present. |  |
| playtime_at_review | number | minutes | Steam review endpoint, frozen language reviews.jsonl | none | Author playtime recorded when the review was submitted. | This is explicitly minutes; convert before interpreting as hours. |
| playtime_forever | number | minutes | Steam review endpoint, frozen language reviews.jsonl | none | Author cumulative playtime at collection time. | May exceed playtime_at_review because play continued after posting. |
| playtime_at_review_hours | number | hours | scripts/clean_reviews.py | Valid non-negative playtime_at_review / 60. | Review-time playtime converted to hours. |  |
| playtime_forever_hours | number | hours | scripts/clean_reviews.py | Valid non-negative playtime_forever / 60. | Cumulative playtime converted to hours. |  |
| log1p_playtime_hours | number | log(1 + hours) | scripts/clean_reviews.py | ln(1 + playtime_at_review / 60). | Log-transformed review-time playtime. |  |
| playtime_p99_group_hours | number | hours | scripts/clean_reviews.py | Linear-interpolated P99 of valid non-negative playtime_at_review hours. | 99th percentile within appid-language group. |  |
| playtime_hours_winsorized_p99 | number | hours | scripts/clean_reviews.py | min(playtime_at_review_hours, playtime_p99_group_hours). | Review-time hours capped at group P99. |  |
| votes_up | integer | votes | Steam review endpoint, frozen language reviews.jsonl | none | Helpful votes received by the review. |  |
| votes_funny | integer | votes | Steam review endpoint, frozen language reviews.jsonl | none | Funny votes received by the review. |  |
| weighted_vote_score | number | Steam score | Steam review endpoint, frozen language reviews.jsonl | none | Steam-provided weighted helpfulness score. | Use as returned; formula is controlled by Steam. |
| comment_count | integer | comments | Steam review endpoint, frozen language reviews.jsonl | none | Number of review comments. |  |
| source | category | none | scripts/steam_data.py | none | Review collection backend. |  |
| game_era | category | none | scripts/steam_data.py | For appid 730, compare creation time with 2023-09-27T00:00:00Z. | CS2/CS:GO date-based era label; other games use other. | appid 730 cumulative playtime may span both CS:GO and CS2. |
| hardware | string | none | Steam review endpoint, frozen language reviews.jsonl | none | Steam hardware field when returned. |  |
| source_relative_path | string | none | scripts/clean_reviews.py | none | Project-relative source JSONL path. |  |
| source_row_number | integer | row number | scripts/clean_reviews.py | none | One-based line number in source JSONL. |  |
| text_key | string | none | scripts/clean_reviews.py | Collapse whitespace, trim, then Unicode casefold. | Key used for exact normalized-text duplicate detection. |  |
| text_length | integer | Unicode code points | scripts/clean_reviews.py | none | Length of text_key. |  |
| english_word_count | integer | tokens | scripts/clean_reviews.py | Regex count of alphabetic words including internal apostrophes. | Count of English alphabetic word tokens. |  |
| chinese_character_count | integer | characters | scripts/clean_reviews.py | Regex count over CJK ideograph ranges. | Count of CJK unified ideograph characters. |  |
| text_duplicate_group_size | integer | reviews | scripts/clean_reviews.py | none | Number of rows sharing appid, language and text_key. |  |
| text_duplicate_rank | integer | rank | scripts/clean_reviews.py | Sort by timestamp_created then recommendationid; first row is canonical. | Deterministic position within a normalized-text duplicate group. |  |
| duplicate_inverse_weight | number | weight | scripts/clean_reviews.py | 1 / text_duplicate_group_size for non-empty text. | Inverse normalized-text group frequency. |  |
| author_review_group_size | integer | reviews | scripts/clean_reviews.py | none | Reviews by the same author within appid-language. |  |
| common_window_start_utc | date | UTC calendar date | scripts/clean_reviews.py | Maximum of the two language-specific earliest valid dates. | Start of the overlapping English/Chinese date window for the game. |  |
| common_window_end_utc | date | UTC calendar date | scripts/clean_reviews.py | Minimum of the two language-specific latest valid dates. | End of the overlapping English/Chinese date window for the game. |  |
| steam_purchase | boolean | none | Steam review endpoint, frozen language reviews.jsonl | none | Steam marks the review as associated with a Steam purchase. |  |
| received_for_free | boolean | none | Steam review endpoint, frozen language reviews.jsonl | none | Steam marks the product as received for free. |  |
| written_during_early_access | boolean | none | Steam review endpoint, frozen language reviews.jsonl | none | Review was written during early access. |  |
| primarily_steam_deck | boolean | none | Steam review endpoint, frozen language reviews.jsonl | none | Steam marks the review as primarily played on Steam Deck. |  |
| refunded | boolean | none | Steam review endpoint, frozen language reviews.jsonl | none | Steam refund indicator when returned. |  |
| is_empty_text | boolean | none | scripts/clean_reviews.py | text_key == ''. | Review text is empty after whitespace normalization. |  |
| is_short_text | boolean | none | scripts/clean_reviews.py | 0 < len(text_key) < 10 in the current build. | Non-empty normalized text is shorter than the configured threshold. |  |
| is_low_information | boolean | none | scripts/clean_reviews.py | English: fewer than 3 alphabetic words; Simplified Chinese: fewer than 4 CJK characters and fewer than 3 English words. | Language-specific low-information text flag. |  |
| is_duplicate_text | boolean | none | scripts/clean_reviews.py | text_duplicate_group_size > 1. | Text belongs to a normalized duplicate group with more than one row. |  |
| is_text_canonical | boolean | none | scripts/clean_reviews.py | First row by timestamp_created then recommendationid. | Row selected to represent a normalized duplicate text group. |  |
| is_duplicate_author_review | boolean | none | scripts/clean_reviews.py | author_review_group_size > 1. | Author has more than one review in the appid-language group. |  |
| is_author_record_canonical | boolean | none | scripts/clean_reviews.py | Latest timestamp_updated, then recommendationid. | Row retained as the canonical review for a repeated author. |  |
| is_invalid_timestamp | boolean | none | scripts/clean_reviews.py | Timestamp parsing failed. | timestamp_created cannot be converted to UTC datetime. |  |
| playtime_at_review_missing | boolean | none | scripts/clean_reviews.py | Numeric parsing returned missing. | Review-time playtime is absent or non-numeric. |  |
| playtime_forever_missing | boolean | none | scripts/clean_reviews.py | Numeric parsing returned missing. | Cumulative playtime is absent or non-numeric. |  |
| playtime_at_review_invalid | boolean | none | scripts/clean_reviews.py | Parsed minutes < 0. | Review-time playtime is negative. |  |
| playtime_forever_invalid | boolean | none | scripts/clean_reviews.py | Parsed minutes < 0. | Cumulative playtime is negative. |  |
| is_playtime_outlier | boolean | none | scripts/clean_reviews.py | playtime_at_review > 5,256,000 minutes (10 years). | Review-time playtime exceeds the configured hard threshold. |  |
| is_playtime_inconsistent | boolean | none | scripts/clean_reviews.py | playtime_at_review > playtime_forever when both are valid. | Review-time playtime exceeds cumulative playtime. |  |
| topic_unique_nonempty | boolean | none | scripts/clean_reviews.py | Canonical author AND non-empty AND canonical text. | Row is a canonical author record with canonical non-empty text. |  |
| topic_fit_eligible | boolean | none | scripts/clean_reviews.py | topic_unique_nonempty AND not low-information. | Row is included in the primary topic-model fitting dataset. |  |
| association_eligible | boolean | none | scripts/clean_reviews.py | Canonical author record AND non-empty text; duplicate text remains with a weight. | Row is included in recommendation/theme association data. |  |
| playtime_analysis_eligible | boolean | none | scripts/clean_reviews.py | association_eligible with non-missing, non-negative, consistent and below-hard-limit playtime. | Row has valid playtime for RQ4 analyses. |  |
| in_common_language_window | boolean | none | scripts/clean_reviews.py | created_date_utc between common window endpoints, inclusive. | Review date lies in its game's English/Chinese overlap window. |  |
| language_comparison_game_eligible | boolean | none | scripts/clean_reviews.py | At least 100 unique non-empty texts per language in the current build. | Game meets the minimum clean text count in both languages inside the overlap window. |  |
| language_comparison_row_eligible | boolean | none | scripts/clean_reviews.py | association_eligible AND in common window AND eligible game. | Row is usable in the strict language comparison subset. |  |

### quality_table

| Field | Type | Unit | Source | Transformation | Description | Caveat |
| --- | --- | --- | --- | --- | --- | --- |
| game | string | none | scripts/verify_data.py over the frozen review JSONL files | Audit aggregation from raw review rows. | Configured game name. |  |
| appid | integer | none | scripts/verify_data.py over the frozen review JSONL files | Audit aggregation from raw review rows. | Steam application identifier. |  |
| language | category | none | scripts/verify_data.py over the frozen review JSONL files | Audit aggregation from raw review rows. | Requested review language. |  |
| raw_review_count | integer | reviews | scripts/verify_data.py over the frozen review JSONL files | Audit aggregation from raw review rows. | Rows collected for the game-language group. |  |
| source_total_matching | integer | reviews | scripts/verify_data.py over the frozen review JSONL files | Audit aggregation from raw review rows. | Steam-reported matching reviews for the query. |  |
| recommendation_count | integer | reviews | scripts/verify_data.py over the frozen review JSONL files | Audit aggregation from raw review rows. | Rows with voted_up=true. |  |
| recommendation_rate | number | proportion | scripts/verify_data.py over the frozen review JSONL files | Audit aggregation from raw review rows. | recommendation_count / raw_review_count. |  |
| earliest_review_date_utc | date | UTC calendar date | scripts/verify_data.py over the frozen review JSONL files | Audit aggregation from raw review rows. | Earliest valid review creation date. |  |
| latest_review_date_utc | date | UTC calendar date | scripts/verify_data.py over the frozen review JSONL files | Audit aggregation from raw review rows. | Latest valid review creation date. |  |
| actual_coverage_days | integer | inclusive UTC days | scripts/verify_data.py over the frozen review JSONL files | Audit aggregation from raw review rows. | Calendar-day span from earliest through latest review. |  |
| empty_text_count | integer | reviews | scripts/verify_data.py over the frozen review JSONL files | Audit aggregation from raw review rows. | Reviews empty after whitespace normalization. |  |
| short_text_count | integer | reviews | scripts/verify_data.py over the frozen review JSONL files | Audit aggregation from raw review rows. | Non-empty normalized reviews shorter than 10 characters. |  |
| duplicate_text_count | integer | extra reviews | scripts/verify_data.py over the frozen review JSONL files | Audit aggregation from raw review rows. | Extra non-empty reviews after appid-language case/whitespace normalization. |  |
| playtime_missing_rate | number | proportion | scripts/verify_data.py over the frozen review JSONL files | Audit aggregation from raw review rows. | Missing/non-numeric playtime_at_review divided by raw_review_count. |  |
| playtime_zero_rate | number | proportion | scripts/verify_data.py over the frozen review JSONL files | Audit aggregation from raw review rows. | Zero playtime_at_review divided by raw_review_count. |  |
| playtime_p25_hours | number | hours | scripts/verify_data.py over the frozen review JSONL files | Audit aggregation from raw review rows. | P25 of valid non-negative playtime_at_review. |  |
| playtime_p50_hours | number | hours | scripts/verify_data.py over the frozen review JSONL files | Audit aggregation from raw review rows. | P50 of valid non-negative playtime_at_review. |  |
| playtime_p75_hours | number | hours | scripts/verify_data.py over the frozen review JSONL files | Audit aggregation from raw review rows. | P75 of valid non-negative playtime_at_review. |  |
| playtime_p90_hours | number | hours | scripts/verify_data.py over the frozen review JSONL files | Audit aggregation from raw review rows. | P90 of valid non-negative playtime_at_review. |  |
| playtime_p95_hours | number | hours | scripts/verify_data.py over the frozen review JSONL files | Audit aggregation from raw review rows. | P95 of valid non-negative playtime_at_review. |  |
| playtime_p99_hours | number | hours | scripts/verify_data.py over the frozen review JSONL files | Audit aggregation from raw review rows. | P99 of valid non-negative playtime_at_review. |  |
| unique_author_count | integer | authors | scripts/verify_data.py over the frozen review JSONL files | Audit aggregation from raw review rows. | Distinct non-empty Steam author IDs. |  |
| duplicate_author_count | integer | authors | scripts/verify_data.py over the frozen review JSONL files | Audit aggregation from raw review rows. | Authors contributing more than one review in the group. |  |
| duplicate_author_review_count | integer | extra reviews | scripts/verify_data.py over the frozen review JSONL files | Audit aggregation from raw review rows. | Reviews beyond one per repeated author. |  |
| core_endpoint_status | category | none | scripts/verify_data.py over the frozen review JSONL files | Audit aggregation from raw review rows. | Combined players/news/store endpoint status. |  |
| review_endpoint_status | category | none | scripts/verify_data.py over the frozen review JSONL files | Audit aggregation from raw review rows. | Review endpoint status for the language. |  |
| endpoint_status | category | none | scripts/verify_data.py over the frozen review JSONL files | Audit aggregation from raw review rows. | Combined core and review endpoint status. |  |
| target_count | integer | reviews | scripts/verify_data.py over the frozen review JSONL files | Audit aggregation from raw review rows. | Requested reviews per language. |  |
| target_shortfall | integer | reviews | scripts/verify_data.py over the frozen review JSONL files | Audit aggregation from raw review rows. | max(target_count - raw_review_count, 0). |  |
| target_met | boolean | none | scripts/verify_data.py over the frozen review JSONL files | Audit aggregation from raw review rows. | Whether raw_review_count reaches target_count. |  |
| met_200_target | boolean | none | scripts/verify_data.py over the frozen review JSONL files | Audit aggregation from raw review rows. | Whether raw_review_count reaches 200. |  |
| directory | string | none | scripts/verify_data.py over the frozen review JSONL files | Audit aggregation from raw review rows. | Source game-language directory. |  |

### game_meta

| Field | Type | Unit | Source | Transformation | Description | Caveat |
| --- | --- | --- | --- | --- | --- | --- |
| appid | integer | none | config/games_clustering.json | Flatten or derive from the stated frozen source. | Steam application identifier. |  |
| game | string | none | freeze manifest and per-run manifest | Flatten or derive from the stated frozen source. | Frozen configured game name. |  |
| selected_run | string | none | freeze manifest and per-run manifest | Flatten or derive from the stated frozen source. | Selected collection run identifier. |  |
| source_directory | string | none | freeze manifest and per-run manifest | Flatten or derive from the stated frozen source. | Project-relative frozen run directory. |  |
| sample_status | category | none | data/quality_table.csv | Flatten or derive from the stated frozen source. | complete when both languages meet target; otherwise partial. | partial means insufficient available reviews in at least one language, not necessarily an endpoint failure. |
| partial_languages | string list | none | data/quality_table.csv | Flatten or derive from the stated frozen source. | Pipe-separated languages below target_count. |  |
| partial_reason | string | none | data/quality_table.csv | Flatten or derive from the stated frozen source. | Observed counts and targets explaining partial status. |  |
| requested_per_language | integer | reviews | freeze manifest and per-run manifest | Flatten or derive from the stated frozen source. | Requested review target for each language. |  |
| english_review_count | integer | reviews | data/quality_table.csv | Flatten or derive from the stated frozen source. | Collected English reviews. |  |
| schinese_review_count | integer | reviews | data/quality_table.csv | Flatten or derive from the stated frozen source. | Collected Simplified Chinese reviews. |  |
| total_review_count | integer | reviews | freeze manifest and per-run manifest | Flatten or derive from the stated frozen source. | Total English plus Simplified Chinese reviews. |  |
| english_source_total_matching | integer | reviews | data/quality_table.csv | Flatten or derive from the stated frozen source. | Steam-reported English matches. |  |
| schinese_source_total_matching | integer | reviews | data/quality_table.csv | Flatten or derive from the stated frozen source. | Steam-reported Simplified Chinese matches. |  |
| english_target_shortfall | integer | reviews | data/quality_table.csv | Flatten or derive from the stated frozen source. | English target shortfall. |  |
| schinese_target_shortfall | integer | reviews | data/quality_table.csv | Flatten or derive from the stated frozen source. | Simplified Chinese target shortfall. |  |
| partial_shortfall_total | integer | reviews | data/quality_table.csv | Flatten or derive from the stated frozen source. | Sum of language target shortfalls. |  |
| english_review_endpoint_status | category | none | data/quality_table.csv | Flatten or derive from the stated frozen source. | English review endpoint audit status. |  |
| schinese_review_endpoint_status | category | none | data/quality_table.csv | Flatten or derive from the stated frozen source. | Simplified Chinese review endpoint audit status. |  |
| sampling_method | string | none | freeze manifest and per-run manifest | Flatten or derive from the stated frozen source. | Collection sampling description. |  |
| sampling_window_start_utc | datetime | UTC | freeze manifest and per-run manifest | Flatten or derive from the stated frozen source. | Frozen sampling-window start. |  |
| sampling_window_end_utc | datetime | UTC | freeze manifest and per-run manifest | Flatten or derive from the stated frozen source. | Frozen sampling-window end. |  |
| sampling_window_days | integer | days | freeze manifest and per-run manifest | Flatten or derive from the stated frozen source. | Configured sampling-window length. |  |
| common_window_start_utc | date | UTC calendar date | cleaning_summary.json | Flatten or derive from the stated frozen source. | English/Chinese overlap-window start. |  |
| common_window_end_utc | date | UTC calendar date | cleaning_summary.json | Flatten or derive from the stated frozen source. | English/Chinese overlap-window end. |  |
| english_unique_nonempty_common_window_count | integer | reviews | cleaning_summary.json | Flatten or derive from the stated frozen source. | Unique non-empty English texts in common window. |  |
| schinese_unique_nonempty_common_window_count | integer | reviews | cleaning_summary.json | Flatten or derive from the stated frozen source. | Unique non-empty Chinese texts in common window. |  |
| language_comparison_eligible | boolean | none | cleaning_summary.json | Flatten or derive from the stated frozen source. | Both languages meet the configured common-window threshold. |  |
| project_genres | string list | none | config/games_clustering.json | Flatten or derive from the stated frozen source. | Project-curated genres used for selection and interpretation. |  |
| project_primary_genre | string | none | config/games_clustering.json | Flatten or derive from the stated frozen source. | First project-curated genre; not an official Steam primary genre. | Project convention only; Steam does not designate the first genre as primary. |
| steam_genres | string list | none | frozen Steam Store appdetails store.json | Flatten or derive from the stated frozen source. | Pipe-separated official Steam Store genre descriptions. |  |
| steam_genre_ids | string list | none | frozen Steam Store appdetails store.json | Flatten or derive from the stated frozen source. | Pipe-separated Steam genre IDs. |  |
| steam_categories | string list | none | frozen Steam Store appdetails store.json | Flatten or derive from the stated frozen source. | Pipe-separated Steam feature/category descriptions. |  |
| steam_category_ids | string list | none | frozen Steam Store appdetails store.json | Flatten or derive from the stated frozen source. | Pipe-separated Steam category IDs. |  |
| store_type | category | none | frozen Steam Store appdetails store.json | Flatten or derive from the stated frozen source. | Steam Store app type. |  |
| is_free | boolean | none | frozen Steam Store appdetails store.json | Flatten or derive from the stated frozen source. | Steam Store free-to-play indicator. |  |
| price_label | string | none | config/games_clustering.json | Flatten or derive from the stated frozen source. | Project configuration price label. |  |
| developers | string list | none | frozen Steam Store appdetails store.json | Flatten or derive from the stated frozen source. | Pipe-separated Steam Store developers. |  |
| publishers | string list | none | frozen Steam Store appdetails store.json | Flatten or derive from the stated frozen source. | Pipe-separated Steam Store publishers. |  |
| release_date_store_text | string | none | frozen Steam Store appdetails store.json | Flatten or derive from the stated frozen source. | Release date text exactly as returned by Steam. |  |
| release_date_iso | date | calendar date | frozen Steam Store appdetails store.json | Flatten or derive from the stated frozen source. | Parsed release date in YYYY-MM-DD when recognized. |  |
| release_date_parse_status | category | none | frozen Steam Store appdetails store.json | Flatten or derive from the stated frozen source. | ok, missing, or unparsed release-date status. |  |
| coming_soon | boolean | none | frozen Steam Store appdetails store.json | Flatten or derive from the stated frozen source. | Steam Store coming-soon flag. |  |
| platform_windows | boolean | none | frozen Steam Store appdetails store.json | Flatten or derive from the stated frozen source. | Steam Store Windows support flag. |  |
| platform_mac | boolean | none | frozen Steam Store appdetails store.json | Flatten or derive from the stated frozen source. | Steam Store macOS support flag. |  |
| platform_linux | boolean | none | frozen Steam Store appdetails store.json | Flatten or derive from the stated frozen source. | Steam Store Linux support flag. |  |
| metacritic_score | integer | score | frozen Steam Store appdetails store.json | Flatten or derive from the stated frozen source. | Metacritic score when supplied by Steam. | Snapshot value; may change after collection. |
| store_recommendations_total | integer | recommendations | frozen Steam Store appdetails store.json | Flatten or derive from the stated frozen source. | Store-level recommendation total at snapshot time. | Snapshot value; may change after collection. |
| store_fetched_at_utc | datetime | UTC | frozen Steam Store appdetails store.json | Flatten or derive from the stated frozen source. | Store snapshot retrieval time. |  |
| store_endpoint | string | none | frozen Steam Store appdetails store.json | Flatten or derive from the stated frozen source. | Steam Store appdetails endpoint URL. |  |
| store_endpoint_status | category | none | freeze manifest and per-run manifest | Flatten or derive from the stated frozen source. | Per-run store endpoint status. |  |
| store_response_success | boolean | none | frozen Steam Store appdetails store.json | Flatten or derive from the stated frozen source. | Final Steam Store response success flag. |  |
| store_fallback_cc | category | country code | freeze manifest and per-run manifest | Flatten or derive from the stated frozen source. | Fallback country code used after the initial store response failed. | Path of Exile used cc=us after cc=cn returned unsuccessful; final store response was successful. |
| store_metadata_source | string | none | frozen Steam Store appdetails store.json | Flatten or derive from the stated frozen source. | Provenance label for metadata values. |  |
| store_metadata_note | string | none | freeze manifest and per-run manifest | Flatten or derive from the stated frozen source. | Material endpoint or provenance note. |  |
| config_verification_status | category | none | config/games_clustering.json | Flatten or derive from the stated frozen source. | Configuration-time store verification status. |  |
| name_match | boolean | none | config/games_clustering.json | Flatten or derive from the stated frozen source. | Configured name matches verified Steam Store name. |  |

