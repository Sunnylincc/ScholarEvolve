from types import SimpleNamespace

from fastapi.testclient import TestClient

import app.main as main_module


class DummyRetriever:
    def is_ready(self) -> bool:
        return True


class DummyRecommendationService:
    def recommend(self, _req):
        return ([{"paper_id": "p1", "score": 0.9, "reasons": {"semantic": 0.9, "final": 0.9}}], 1.0)


def test_health() -> None:
    dummy_state = SimpleNamespace(
        paper_lookup={
            "p1": SimpleNamespace(
                paper_id="p1",
                title="t",
                abstract="a",
                authors=["x"],
                categories=["cs.AI"],
                published_date="2024-01-01T00:00:00+00:00",
                metadata={},
            )
        },
        rust=SimpleNamespace(module=True),
        retriever=DummyRetriever(),
        recommendation_service=DummyRecommendationService(),
    )

    main_module._state = dummy_state  # type: ignore[assignment]
    client = TestClient(main_module.app)
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
