"""Tests for FRC field geometry and zone lookups."""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from frc_rl.env.field import (
    FRCField,
    GamePiece,
    FIELD_LENGTH,
    FIELD_WIDTH,
)


@pytest.fixture
def field() -> FRCField:
    return FRCField.reefscape_2025()


class TestFieldBoundaries:
    def test_clamp_inside(self):
        x, y = FRCField.clamp_to_field(5.0, 3.0)
        assert x == pytest.approx(5.0)
        assert y == pytest.approx(3.0)

    def test_clamp_negative(self):
        x, y = FRCField.clamp_to_field(-1.0, -2.0)
        assert x == pytest.approx(0.0)
        assert y == pytest.approx(0.0)

    def test_clamp_over_max(self):
        x, y = FRCField.clamp_to_field(FIELD_LENGTH + 5, FIELD_WIDTH + 5)
        assert x == pytest.approx(FIELD_LENGTH)
        assert y == pytest.approx(FIELD_WIDTH)

    def test_out_of_bounds_true(self):
        assert FRCField.is_out_of_bounds(-0.1, 4.0) is True

    def test_out_of_bounds_false(self):
        assert FRCField.is_out_of_bounds(5.0, 4.0) is False

    def test_wall_contact_at_edge(self):
        assert FRCField.wall_contact(0.0, 4.0) is True
        assert FRCField.wall_contact(FIELD_LENGTH, 4.0) is True

    def test_wall_contact_inside(self):
        assert FRCField.wall_contact(5.0, 4.0) is False


class TestScoringZones:
    def test_has_blue_and_red_zones(self, field):
        blue = field.get_scoring_zones_for_alliance("blue")
        red = field.get_scoring_zones_for_alliance("red")
        assert len(blue) >= 4
        assert len(red) >= 4

    def test_find_scoring_zone_hit(self, field):
        # Place robot directly on a blue reef zone
        zone = field.scoring_zones[5]  # first blue zone
        result = field.find_scoring_zone(zone.x, zone.y, "blue")
        assert result is not None
        assert result.alliance == "blue"

    def test_find_scoring_zone_miss(self, field):
        # Far away from any zone
        result = field.find_scoring_zone(8.0, 4.0, "blue")
        assert result is None

    def test_blue_robot_cannot_score_in_red_zone(self, field):
        # Red reef at ~13.9
        result = field.find_scoring_zone(13.90, 4.10, "blue")
        assert result is None


class TestIntakeZones:
    def test_find_intake_zone(self, field):
        zone = field.intake_zones[0]
        result = field.find_intake_zone(zone.x, zone.y, zone.alliance)
        assert result is not None

    def test_wrong_alliance_intake_zone(self, field):
        # Red station for blue robot
        red_zone = next(z for z in field.intake_zones if z.alliance == "red")
        result = field.find_intake_zone(red_zone.x, red_zone.y, "blue")
        assert result is None


class TestGamePieces:
    def test_nearest_game_piece(self, field):
        pieces = [GamePiece(0, 5.0, 3.0), GamePiece(1, 10.0, 5.0)]
        nearest = field.nearest_game_piece(5.5, 3.0, pieces)
        assert nearest is not None
        assert nearest.piece_id == 0

    def test_nearest_skips_inactive(self, field):
        pieces = [
            GamePiece(0, 5.0, 3.0, active=False),
            GamePiece(1, 10.0, 5.0, active=True),
        ]
        nearest = field.nearest_game_piece(5.5, 3.0, pieces)
        assert nearest is not None
        assert nearest.piece_id == 1

    def test_nearest_empty_list(self, field):
        result = field.nearest_game_piece(5.0, 3.0, [])
        assert result is None


class TestStartingPoses:
    def test_blue_poses_near_left_wall(self, field):
        poses = FRCField.starting_poses("blue")
        assert len(poses) == 3
        for x, y, theta in poses:
            assert x < FIELD_LENGTH / 2
            assert math.isclose(theta, 0.0, abs_tol=0.01)

    def test_red_poses_near_right_wall(self, field):
        poses = FRCField.starting_poses("red")
        assert len(poses) == 3
        for x, y, theta in poses:
            assert x > FIELD_LENGTH / 2
