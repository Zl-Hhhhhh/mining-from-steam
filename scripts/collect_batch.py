"""Batch collection script for multiple games from games_clustering.json."""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

# Import from steam_data.py
from steam_data import collect as collect_single_game, save_json


def load_games_config():
    """Load games list from games_clustering.json."""
    config_path = ROOT / "config" / "games_clustering.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Games config not found: {config_path}")
    
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return config


def load_progress():
    """Load progress from checkpoint file."""
    checkpoint = ROOT / "data" / "batch_progress.json"
    if checkpoint.exists():
        return json.loads(checkpoint.read_text(encoding="utf-8"))
    return {"completed": [], "failed": [], "last_updated": None}


def save_progress(progress):
    """Save progress to checkpoint file."""
    checkpoint = ROOT / "data" / "batch_progress.json"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    progress["last_updated"] = datetime.now(timezone.utc).isoformat()
    save_json(checkpoint, progress)


def collect_batch(args):
    """Batch collect reviews for multiple games."""
    config = load_games_config()
    games = config["games"]
    languages = config["collection_config"]["languages"]
    reviews_per_language = config["collection_config"]["reviews_per_language"]
    
    progress = load_progress()
    
    # Filter games based on resume mode
    if args.resume:
        completed_appids = {g["appid"] for g in progress["completed"]}
        games_to_collect = [g for g in games if g["appid"] not in completed_appids]
        print(f"Resume mode: {len(completed_appids)} games already completed")
        print(f"Remaining: {len(games_to_collect)} games")
    else:
        games_to_collect = games
        if not args.force and progress["completed"]:
            print(f"Warning: Found existing progress with {len(progress['completed'])} completed games")
            print("Use --resume to continue, or --force to start fresh")
            return 1
    
    # Apply limit if specified
    if args.limit:
        games_to_collect = games_to_collect[:args.limit]
    
    # Statistics
    total_games = len(games_to_collect)
    successful = 0
    failed = 0
    start_time = time.time()
    
    print(f"\n{'='*70}")
    print(f"Batch Collection Started")
    print(f"{'='*70}")
    print(f"Total games to collect: {total_games}")
    print(f"Reviews per game: {reviews_per_language} × {len(languages)} languages = {reviews_per_language * len(languages)}")
    print(f"Languages: {', '.join(languages)}")
    print(f"Estimated time: ~{total_games * 4} minutes ({total_games * 4 / 60:.1f} hours)")
    print(f"{'='*70}\n")
    
    for idx, game in enumerate(games_to_collect, 1):
        appid = game["appid"]
        name = game["name"]
        
        print(f"\n[{idx}/{total_games}] Collecting: {name} (appid: {appid})")
        print(f"  Genre: {', '.join(game['genre'][:3])}")
        print(f"  Progress: {successful} successful, {failed} failed")
        
        # Build arguments for single game collection
        class GameArgs:
            pass
        
        game_args = GameArgs()
        game_args.appid = appid
        game_args.languages = languages
        game_args.limit = reviews_per_language
        game_args.backend = "official"
        game_args.max_pages = 50  # Safety limit
        game_args.delay = 1.5  # Rate limiting
        
        game_start = time.time()
        
        try:
            # Call the original collect function
            exit_code = collect_single_game(game_args)
            
            game_elapsed = time.time() - game_start
            
            if exit_code == 0:
                print(f"  ✅ SUCCESS ({game_elapsed:.1f}s)")
                successful += 1
                progress["completed"].append({
                    "appid": appid,
                    "name": name,
                    "collected_at": datetime.now(timezone.utc).isoformat(),
                    "elapsed_seconds": game_elapsed
                })
            else:
                print(f"  ⚠️  PARTIAL ({game_elapsed:.1f}s) - some endpoints failed")
                successful += 1  # Count as success but note the issue
                progress["completed"].append({
                    "appid": appid,
                    "name": name,
                    "collected_at": datetime.now(timezone.utc).isoformat(),
                    "elapsed_seconds": game_elapsed,
                    "note": "partial_success"
                })
            
            # Save progress after each successful collection
            save_progress(progress)
            
        except KeyboardInterrupt:
            print(f"\n\n⚠️  Interrupted by user")
            print(f"Progress saved. Use --resume to continue later.")
            save_progress(progress)
            return 1
            
        except Exception as error:
            game_elapsed = time.time() - game_start
            print(f"  ❌ FAILED ({game_elapsed:.1f}s): {error}")
            failed += 1
            progress["failed"].append({
                "appid": appid,
                "name": name,
                "failed_at": datetime.now(timezone.utc).isoformat(),
                "error": str(error)
            })
            save_progress(progress)
        
        # Inter-game delay to be respectful to Steam API
        if idx < total_games:
            delay = args.game_delay
            print(f"  Waiting {delay}s before next game...")
            time.sleep(delay)
        
        # Progress summary
        elapsed = time.time() - start_time
        remaining = total_games - idx
        avg_time_per_game = elapsed / idx
        estimated_remaining = remaining * avg_time_per_game
        
        print(f"  Elapsed: {elapsed/60:.1f}min | ETA: {estimated_remaining/60:.1f}min")
    
    # Final summary
    total_elapsed = time.time() - start_time
    
    print(f"\n{'='*70}")
    print(f"Batch Collection Complete")
    print(f"{'='*70}")
    print(f"Successful: {successful}/{total_games}")
    print(f"Failed: {failed}/{total_games}")
    print(f"Total time: {total_elapsed/60:.1f} minutes ({total_elapsed/3600:.2f} hours)")
    print(f"Average per game: {total_elapsed/total_games:.1f} seconds")
    print(f"{'='*70}")
    
    # Generate summary report
    summary = {
        "collection_type": "batch",
        "started_at": datetime.fromtimestamp(start_time, timezone.utc).isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "total_games": total_games,
        "successful": successful,
        "failed": failed,
        "elapsed_seconds": total_elapsed,
        "config_file": "config/games_clustering.json",
        "reviews_per_language": reviews_per_language,
        "languages": languages,
        "completed_games": progress["completed"][-successful:] if successful else [],
        "failed_games": progress["failed"][-failed:] if failed else []
    }
    
    summary_path = ROOT / "data" / "batch_summary.json"
    save_json(summary_path, summary)
    print(f"\nSummary saved to: {summary_path.relative_to(ROOT)}")
    
    # Clean up checkpoint if all completed
    if failed == 0 and not args.resume:
        checkpoint = ROOT / "data" / "batch_progress.json"
        if checkpoint.exists():
            checkpoint.unlink()
            print("Checkpoint file cleaned up (all games successful)")
    
    return 0 if failed == 0 else 1


def main():
    parser = argparse.ArgumentParser(
        description="Batch collect Steam reviews for multiple games",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Collect all 50 games
  python scripts/collect_batch.py
  
  # Test with first 3 games
  python scripts/collect_batch.py --limit 3
  
  # Resume after interruption
  python scripts/collect_batch.py --resume
  
  # Start fresh (ignore existing progress)
  python scripts/collect_batch.py --force
  
  # Increase delay between games (default 3s)
  python scripts/collect_batch.py --game-delay 5
"""
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        help="Only collect first N games (for testing)"
    )
    
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint (skip completed games)"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Start fresh, ignore existing progress"
    )
    
    parser.add_argument(
        "--game-delay",
        type=float,
        default=3.0,
        help="Delay in seconds between games (default: 3.0)"
    )
    
    args = parser.parse_args()
    
    try:
        sys.exit(collect_batch(args))
    except Exception as error:
        print(f"\n❌ Fatal error: {error}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
