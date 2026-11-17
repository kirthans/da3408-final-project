"""Promotion preserves a useful rollback target and validates before alias writes."""

from types import SimpleNamespace

import pytest

from elo.training import registry


@pytest.fixture
def client(monkeypatch):
    class Registry:
        aliases = {"champion": "2"}

        def delete_registered_model_alias(self, name, alias):
            self.aliases.pop(alias)

        def set_model_version_tag(self, name, version, key, value):
            self.tag = (version, key, value)

        def get_model_version(self, name, version):
            if version not in {"1", "2", "3"}:
                raise ValueError("Unknown version")

        def get_registered_model(self, name):
            return SimpleNamespace(aliases=self.aliases.copy())

        def set_registered_model_alias(self, name, alias, version):
            self.aliases[alias] = version

    client = Registry()
    monkeypatch.setattr(registry, "MlflowClient", lambda: client)
    return client


def test_promote_repeat_and_rollback(client):
    registry.promote(3)
    assert client.aliases == {"champion": "3", "previous_champion": "2"}
    registry.promote(3)
    assert client.aliases == {"champion": "3", "previous_champion": "2"}
    registry.rollback()
    assert client.aliases["champion"] == "2"
    registry.promote(3)
    assert client.aliases == {"champion": "3", "previous_champion": "2"}


def test_invalid_promotion_leaves_aliases_unchanged(client):
    with pytest.raises(ValueError, match="Unknown version"):
        registry.promote(99)
    assert client.aliases == {"champion": "2"}


def test_initial_promotion_and_missing_rollback(client):
    client.aliases.clear()
    registry.rollback()
    assert client.aliases == {}
    registry.promote(1)
    assert client.aliases == {"champion": "1"}


def test_challenger_cleared_on_promote_and_reject(client):
    client.aliases["challenger"] = "3"
    registry.promote(3)
    assert "challenger" not in client.aliases
    client.aliases["challenger"] = "1"
    registry.reject(1)
    assert "challenger" not in client.aliases
    assert client.tag == ("1", "gate", "rejected")
    assert client.aliases["champion"] == "3"


def test_reset_removes_replay_aliases_even_when_baseline_is_champion(client):
    client.aliases.update(previous_champion="3", challenger="1")
    registry.reset(1)
    assert client.aliases == {"champion": "1"}
    client.aliases.update(previous_champion="3", challenger="2")
    registry.reset(1)
    assert client.aliases == {"champion": "1"}
