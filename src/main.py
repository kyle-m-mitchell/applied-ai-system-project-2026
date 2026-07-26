"""
Command line runner for the Music Recommender Simulation.

Runs the functional path end to end:
    validate request -> RecommendationService -> original scoring core
"""

from src.contracts import RecommendationRequest, RecommendationResult
from src.recommender import load_songs
from src.service import RecommendationService


def format_request(request: RecommendationRequest) -> str:
    """Render supplied preferences as a compact, single-line summary."""
    values = request.model_dump(exclude_none=True, exclude={"limit"})
    return ", ".join(f"{key}={value}" for key, value in values.items())


def print_recommendations(result: RecommendationResult) -> None:
    """Print recommendations in a clean, readable terminal layout."""
    divider = "-" * 64
    print("\n🎵  Music Recommender — your top picks\n")
    print(f"Taste profile: {format_request(result.request)}")
    print(f"Operating mode: {result.operating_mode.value}")
    print(divider)
    for rank, item in enumerate(result.recommendations, start=1):
        track = item.track
        print(
            f"{rank}. {track.title} — {track.artist}  "
            f"[{track.genre} · {track.mood}]"
        )
        print(
            f"   Raw score: {item.raw_score:.2f}  ·  "
            f"Match strength: {item.match_strength:.0%}"
        )
        print("   Why:")
        for reason in item.reasons:
            print(f"     • {reason}")
        print(divider)

    for warning in result.warnings:
        print(f"Note: {warning}")


def main() -> None:
    songs = load_songs("data/songs.csv")
    service = RecommendationService(songs)

    request = RecommendationRequest(
        genre="lofi",
        mood="chill",
        energy=0.40,
        acousticness=0.80,
        valence=0.55,
        danceability=0.40,
        tempo_bpm=78,  # A relaxed study-beat tempo.
        limit=5,
    )

    result = service.recommend(request)
    print_recommendations(result)


if __name__ == "__main__":
    main()
