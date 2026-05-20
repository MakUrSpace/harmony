"""
Tests targeting uncovered user input logic flows in harmonyServer.py and configurator.py.
Focus: save/load, object rename/delete, session management, VirtualMap selection, move requests.
"""

import pytest
import json
import os
import pickle
import tempfile
import unittest.mock as mock
import numpy as np


# ---------------------------------------------------------------------------
# Harmony server helpers
# ---------------------------------------------------------------------------


def _setup_session(hs, view_id, **kwargs):
    from harmony.harmonyServer import SessionConfig

    s = SessionConfig(**kwargs)
    hs.SESSIONS[view_id] = s
    return s


def _mock_obj(oid, axial=(5, 5)):
    obj = mock.MagicMock()
    obj.oid = oid
    obj.constituent_axials = [axial]
    del obj.objectType
    return obj


# ---------------------------------------------------------------------------
# get_conversion_params / get_virtual_map_crop
# ---------------------------------------------------------------------------


class TestConversionParams:
    def test_virtual_map_no_cm(self, harmony_client):
        import harmony.harmonyServer as hs

        orig = hs._cm
        hs._cm = None
        sx, sy, lx, ly = hs.get_conversion_params("VirtualMap")
        assert sx == 1.0 and sy == 1.0
        hs._cm = orig

    def test_camera_missing(self, harmony_client):
        import harmony.harmonyServer as hs

        hs._cc.cameras = {}
        sx, sy, lx, ly = hs.get_conversion_params("Nonexistent")
        assert sx == 1.0

    def test_camera_with_bounding_box(self, harmony_client):
        import harmony.harmonyServer as hs

        hs._cc.cameras["Camera 0"].activeZoneBoundingBox = (10, 20, 200, 100)
        sx, sy, lx, ly = hs.get_conversion_params("Camera 0")
        assert sx > 0 and sy > 0

    def test_virtual_map_with_bounding_box(self, harmony_client):
        import harmony.harmonyServer as hs

        hs._cm.cc.realSpaceBoundingBox.return_value = (0, 0, 400, 300)
        hs._cm.cc.hex = mock.MagicMock()
        hs._cm.cc.hex.width = 800
        hs._cm.cc.hex.height = 800
        sx, sy, lx, ly = hs.get_conversion_params("VirtualMap")
        assert sx > 0


# ---------------------------------------------------------------------------
# save / load game
# ---------------------------------------------------------------------------


class TestSaveLoad:
    def test_save_game_get(self, harmony_client):
        import harmony.harmonyServer as hs

        hs._cm.memory = []
        with (
            mock.patch.object(hs._cm, "saveGame"),
            mock.patch("builtins.open", mock.mock_open()),
            mock.patch("pickle.dump"),
        ):
            r = harmony_client.get("/harmony/save_game/testgame_save")
        assert r.status_code == 200
        assert b"saved" in r.content.lower()

    def test_load_game_missing_file(self, harmony_client):
        r = harmony_client.get("/harmony/load_game/nonexistent_game_xyz")
        assert r.status_code == 200
        assert b"not found" in r.content.lower()

    def test_load_game_existing(self, harmony_client):
        import harmony.harmonyServer as hs
        from harmony.harmonyServer import SessionConfig

        load_data = {"memory": [], "sessions": {"s1": SessionConfig()}}
        with tempfile.NamedTemporaryFile(
            suffix=".pickle", delete=False, dir=os.getcwd(), prefix="testgame_"
        ) as f:
            pickle.dump(load_data, f)
            game_name = os.path.basename(f.name).replace(".pickle", "")
        try:
            r = harmony_client.get(f"/harmony/load_game/{game_name}")
            assert r.status_code == 200
            assert b"loaded" in r.content.lower()
        finally:
            os.unlink(f.name)

    def test_save_post(self, harmony_client):
        import harmony.harmonyServer as hs

        with tempfile.TemporaryDirectory() as td:
            game = os.path.join(td, "postgame")
            with mock.patch.object(hs._cm, "saveGame"):
                r = harmony_client.post("/harmony/save", data={"game_name": game})
            assert r.status_code == 200

    def test_save_post_no_name(self, harmony_client):
        r = harmony_client.post("/harmony/save", data={"game_name": ""})
        assert r.status_code in (200, 400)

    def test_load_post(self, harmony_client):
        r = harmony_client.post("/harmony/load", data={"game_name": "nonexistent_xyz"})
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# Session ID management
# ---------------------------------------------------------------------------


class TestSessionManagement:
    def test_update_session_id_rename(self, harmony_client):
        import harmony.harmonyServer as hs
        from harmony.harmonyServer import SessionConfig

        hs.SESSIONS["old123"] = SessionConfig()
        r = harmony_client.post(
            "/harmony/update_session_id",
            data={"viewId": "old123", "newViewId": "new456"},
        )
        assert r.status_code == 200
        assert b"new456" in r.content
        assert "new456" in hs.SESSIONS
        assert "old123" not in hs.SESSIONS

    def test_update_session_id_missing_old(self, harmony_client):
        r = harmony_client.post(
            "/harmony/update_session_id",
            data={"viewId": "does_not_exist", "newViewId": "brand_new"},
        )
        assert r.status_code == 404

    def test_update_session_id_existing_target(self, harmony_client):
        import harmony.harmonyServer as hs
        from harmony.harmonyServer import SessionConfig

        hs.SESSIONS["src"] = SessionConfig()
        hs.SESSIONS["dst"] = SessionConfig()
        r = harmony_client.post(
            "/harmony/update_session_id", data={"viewId": "src", "newViewId": "dst"}
        )
        assert r.status_code == 200
        assert b"dst" in r.content


# ---------------------------------------------------------------------------
# Object CRUD via /harmony/objects/
# ---------------------------------------------------------------------------


class TestObjectCRUD:
    def _inject_obj(self, hs, oid):
        obj = _mock_obj(oid)
        hs._cm.memory = [obj]
        hs._cm.findObject.side_effect = (
            lambda objectId: obj if objectId == oid else None
        )
        return obj

    def test_get_object_not_found(self, harmony_client):
        import harmony.harmonyServer as hs

        hs._cm.findObject.side_effect = lambda oid: None
        r = harmony_client.get("/harmony/objects/ghost_obj")
        assert r.status_code == 404

    def test_get_object_found(self, harmony_client):
        import harmony.harmonyServer as hs

        obj = self._inject_obj(hs, "Hero")
        with (
            mock.patch("harmony.harmonyServer.imageToBase64", return_value="b64data"),
            mock.patch(
                "harmony.harmonyServer.buildObjectSettings",
                return_value="settings_html",
            ),
        ):
            r = harmony_client.get("/harmony/objects/Hero")
        assert r.status_code == 200

    def test_delete_object_admin(self, harmony_client):
        import harmony.harmonyServer as hs
        from harmony.harmonyServer import SessionConfig

        obj = self._inject_obj(hs, "Target")
        hs._cm.deleteObject = mock.MagicMock()
        view_id = "admin_del"
        hs.SESSIONS[view_id] = SessionConfig()
        r = harmony_client.request(
            "DELETE", "/harmony/objects/Target", cookies={"session_view_id": view_id}
        )
        assert r.status_code == 200
        hs._cm.deleteObject.assert_called_once_with("Target")

    def test_post_object_rename(self, harmony_app):
        """Rename succeeds when object exists and new name is free."""
        from fastapi.testclient import TestClient
        import harmony.harmonyServer as hs
        from harmony.harmonyServer import SessionConfig

        obj = _mock_obj("OldName")
        mock_cm = mock.MagicMock()
        mock_cm.memory = [obj]
        # _find_object calls findObject(objectId=oid) with keyword arg
        mock_cm.findObject.side_effect = (
            lambda objectId=None: obj if objectId == "OldName" else None
        )
        mock_cm.cc.trackedObjectLastDistance.return_value = None
        mock_cm.cc.changeSetToAxialCoord.return_value = (0, 0)
        hs._cm = mock_cm
        view_id = "ren_session2"
        hs.SESSIONS[view_id] = SessionConfig(moveable=["OldName"])
        c = TestClient(harmony_app)
        c.cookies.set("session_view_id", view_id)
        with (
            mock.patch("harmony.harmonyServer.imageToBase64", return_value="b64"),
            mock.patch(
                "harmony.harmonyServer.buildObjectTable", return_value="<p>ok</p>"
            ),
        ):
            r = c.post("/harmony/objects/OldName", data={"objectName": "NewName"})
        assert r.status_code == 200
        assert obj.oid == "NewName"

    def test_post_object_rename_duplicate(self, harmony_app):
        """Rename returns 400 when new name is already taken."""
        from fastapi.testclient import TestClient
        import harmony.harmonyServer as hs
        from harmony.harmonyServer import SessionConfig

        obj = _mock_obj("OrigObj2")
        existing = _mock_obj("DupeObj2")
        mock_cm = mock.MagicMock()
        mock_cm.memory = [obj, existing]

        def find(objectId=None):
            if objectId == "OrigObj2":
                return obj
            if objectId == "DupeObj2":
                return existing
            return None

        mock_cm.findObject.side_effect = find
        hs._cm = mock_cm
        view_id = "dupe_ses2"
        hs.SESSIONS[view_id] = SessionConfig(moveable=["OrigObj2"])
        c = TestClient(harmony_app)
        c.cookies.set("session_view_id", view_id)
        r = c.post("/harmony/objects/OrigObj2", data={"objectName": "DupeObj2"})
        assert r.status_code == 400


# ---------------------------------------------------------------------------
# Object factory (define + delete)
# ---------------------------------------------------------------------------


class TestObjectFactory:
    def test_build_object_factory_single_cell(self, harmony_client):
        import harmony.harmonyServer as hs
        from harmony.harmonyServer import SessionConfig, CellSelection

        view_id = "factory1"
        _setup_session(hs, view_id)
        hs.SESSIONS[view_id].selection = CellSelection(firstCell=(1, 2))
        r = harmony_client.get(f"/harmony/object_factory/{view_id}")
        assert r.status_code == 200
        assert b"object_name" in r.content

    def test_post_object_factory_single(self, harmony_client):
        import harmony.harmonyServer as hs
        from harmony.harmonyServer import SessionConfig, CellSelection

        view_id = "factory2"
        _setup_session(hs, view_id)
        hs.SESSIONS[view_id].selection = CellSelection(firstCell=(3, 4))
        new_obj = _mock_obj("Widget")
        hs._cc.define_object_from_axial.return_value = new_obj
        hs._cm.cc = hs._cc
        r = harmony_client.post(
            f"/harmony/object_factory/{view_id}", data={"object_name": "Widget"}
        )
        assert r.status_code == 200
        assert b"Widget" in r.content

    def test_post_object_factory_multi_cell(self, harmony_client):
        import harmony.harmonyServer as hs
        from harmony.harmonyServer import SessionConfig, CellSelection

        view_id = "factory3"
        _setup_session(hs, view_id)
        hs.SESSIONS[view_id].selection = CellSelection(
            firstCell=(0, 0), additionalCells=[(1, 0), (0, 1)]
        )
        new_obj = _mock_obj("MultiObj")
        hs._cc.define_object_from_axials.return_value = new_obj
        hs._cm.cc = hs._cc
        r = harmony_client.post(
            f"/harmony/object_factory/{view_id}", data={"object_name": "MultiObj"}
        )
        assert r.status_code == 200
        assert b"MultiObj" in r.content

    def test_object_factory_forbidden_user(self, harmony_app):
        from fastapi.testclient import TestClient
        import harmony.harmonyServer as hs
        from harmony.harmonyServer import SessionConfig, CellSelection

        user_app = hs.create_harmony_app(template_name="HarmonyUser.html")
        uc = TestClient(user_app)
        view_id = "uf1"
        _setup_session(hs, view_id)
        hs.SESSIONS[view_id].selection = CellSelection(firstCell=(1, 1))
        r = uc.get(f"/harmony/object_factory/{view_id}")
        assert r.status_code == 403


# ---------------------------------------------------------------------------
# Move request
# ---------------------------------------------------------------------------


class TestMoveRequest:
    def test_move_object_success(self, harmony_client):
        import harmony.harmonyServer as hs
        from harmony.harmonyServer import SessionConfig, CellSelection

        obj = _mock_obj("Mover", axial=(2, 2))
        hs._cm.memory = [obj]
        hs._cm.findObject.side_effect = lambda oid: obj if oid == "Mover" else None
        new_obj = _mock_obj("Mover", axial=(4, 4))
        hs._cc.define_object_from_axial.return_value = new_obj
        hs._cm.cc = hs._cc
        view_id = "move1"
        _setup_session(hs, view_id, moveable=["Mover"])
        hs.SESSIONS[view_id].selection = CellSelection(
            firstCell=(2, 2), additionalCells=[(4, 4)]
        )
        r = harmony_client.get(f"/harmony/request_move/Mover/{view_id}")
        assert r.status_code == 200

    def test_move_object_not_moveable(self, harmony_app):
        from fastapi.testclient import TestClient
        import harmony.harmonyServer as hs
        from harmony.harmonyServer import SessionConfig, CellSelection

        # Use the user app — users without moveable permission should get 403
        user_app = hs.create_harmony_app(template_name="HarmonyUser.html")
        uc = TestClient(user_app)
        view_id = "move2"
        _setup_session(hs, view_id, moveable=[])
        hs.SESSIONS[view_id].selection = CellSelection(
            firstCell=(0, 0), additionalCells=[(1, 1)]
        )
        obj = _mock_obj("ImmovableObj")
        hs._cm.findObject.side_effect = (
            lambda oid: obj if oid == "ImmovableObj" else None
        )
        r = uc.get(f"/harmony/request_move/ImmovableObj/{view_id}")
        assert r.status_code == 403

    def test_move_object_not_found(self, harmony_client):
        import harmony.harmonyServer as hs
        from harmony.harmonyServer import SessionConfig, CellSelection

        view_id = "move3"
        _setup_session(hs, view_id, moveable=["Ghost"])
        hs.SESSIONS[view_id].selection = CellSelection(
            firstCell=(0, 0), additionalCells=[(1, 1)]
        )
        hs._cm.findObject.side_effect = lambda oid: None
        r = harmony_client.get(f"/harmony/request_move/Ghost/{view_id}")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# VirtualMap pixel selection
# ---------------------------------------------------------------------------


class TestVirtualMapSelection:
    def test_select_pixel_virtual_map(self, harmony_client):
        import harmony.harmonyServer as hs
        from harmony.harmonyServer import SessionConfig

        hs._cm.cc = hs._cc
        hs._cc.pixel_to_axial.return_value = (3, 7)
        hs._cc.realSpaceBoundingBox.return_value = (0, 0, 400, 300)
        hs._cm.cc.hex_at_axial.return_value = [np.array([100, 100])]
        hs._cm.cc.cam_hex_at_axial.return_value = [np.array([50, 50])]
        hs._cm.memory = []
        view_id = "vm_sel"
        _setup_session(hs, view_id)
        r = harmony_client.post(
            "/harmony/select_pixel",
            data={
                "viewId": view_id,
                "selectedPixel": "[300, 200]",
                "selectedCamera": "VirtualMap",
                "appendPixel": "",
            },
        )
        assert r.status_code == 200
        assert hs.SESSIONS[view_id].selection.firstCell == (3, 7)


# ---------------------------------------------------------------------------
# Toggle selection cycling
# ---------------------------------------------------------------------------


class TestToggleSelection:
    def test_toggle_cycles_objects(self, harmony_client):
        import harmony.harmonyServer as hs
        from harmony.harmonyServer import SessionConfig, CellSelection

        obj_a = _mock_obj("Alpha", (1, 1))
        obj_b = _mock_obj("Beta", (1, 1))
        hs._cm.memory = [obj_a, obj_b]
        hs._cc.changeSetToAxialCoord.side_effect = lambda o: (1, 1)
        hs._cm.cc = hs._cc
        view_id = "cyc1"
        _setup_session(hs, view_id)
        hs.SESSIONS[view_id].selection = CellSelection(firstCell=(1, 1))
        hs.SESSIONS[view_id].selected_oid = "Alpha"
        r = harmony_client.post(f"/harmony/toggle_selection/{view_id}")
        assert r.status_code == 200

    def test_toggle_no_selection(self, harmony_client):
        import harmony.harmonyServer as hs
        from harmony.harmonyServer import SessionConfig

        view_id = "cyc2"
        _setup_session(hs, view_id)
        r = harmony_client.post(f"/harmony/toggle_selection/{view_id}")
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# Select unit in table
# ---------------------------------------------------------------------------


class TestSelectUnit:
    def test_select_unit_by_oid(self, harmony_client):
        import harmony.harmonyServer as hs
        from harmony.harmonyServer import SessionConfig

        obj = _mock_obj("UnitA", (3, 3))
        hs._cm.memory = [obj]
        hs._cm.cc = hs._cc
        hs._cc.hex_at_axial.return_value = [np.array([100, 100])]
        hs._cc.cam_hex_at_axial.return_value = [np.array([50, 50])]
        view_id = "su1"
        _setup_session(hs, view_id)
        r = harmony_client.post(f"/harmony/select_unit/UnitA/{view_id}")
        assert r.status_code == 200
        assert hs.SESSIONS[view_id].selected_oid == "UnitA"

    def test_select_unit_not_found(self, harmony_client):
        import harmony.harmonyServer as hs

        hs._cm.memory = []
        r = harmony_client.post("/harmony/select_unit/GhostUnit/view99")
        assert r.status_code in (200, 404)


# ---------------------------------------------------------------------------
# Reset admin-only
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_forbidden_for_user(self, harmony_app):
        from fastapi.testclient import TestClient
        import harmony.harmonyServer as hs

        user_app = hs.create_harmony_app(template_name="HarmonyUser.html")
        uc = TestClient(user_app)
        r = uc.get("/harmony/reset")
        assert r.status_code == 403

    def test_reset_admin_succeeds(self, harmony_client):
        r = harmony_client.get("/harmony/reset")
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# Configurator-side nudge and camera endpoints
# ---------------------------------------------------------------------------


class TestConfiguratorInputFlows:
    def test_nudge_commit_single_cell(self, configurator_client, configurator_app):
        cc = configurator_app.state.cc
        cc.hex.hex_nudges = {}
        cc.hex.nudge_hex = mock.MagicMock()
        cc.saveConfiguration = mock.MagicMock()
        payload = {
            "hexes": [{"q": 1, "r": 2}],
            "dx": 8,
            "dy": -4,
            "reset": False,
            "reset_all": False,
        }
        r = configurator_client.post("/configurator/nudge/Camera 0", json=payload)
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "success"

    def test_nudge_reset_all(self, configurator_client, configurator_app):
        cc = configurator_app.state.cc
        cc.hex.hex_nudges = {"Camera 0": {"1,2": [5, 5]}}
        cc.saveConfiguration = mock.MagicMock()
        payload = {"hexes": [], "dx": 0, "dy": 0, "reset": False, "reset_all": True}
        r = configurator_client.post("/configurator/nudge/Camera 0", json=payload)
        assert r.status_code == 200
        assert r.json()["status"] == "success"

    def test_nudge_select_existing_nudge(self, configurator_client, configurator_app):
        cc = configurator_app.state.cc
        cc.camCoordToAxial.return_value = (2, 3)
        cc.cam_hex_at_axial.return_value = np.array(
            [[10, 10], [20, 10], [25, 20], [20, 30], [10, 30], [5, 20]]
        )
        cc.hex.hex_nudges = {"Camera 0": {"2,3": [7, -3]}}
        r = configurator_client.get("/configurator/nudge_select/Camera 0?px=0.5&py=0.5")
        assert r.status_code == 200
        data = r.json()
        assert data["existing_dx"] == 7
        assert data["existing_dy"] == -3

    def test_set_overlays(self, configurator_client, configurator_app):
        r = configurator_client.post(
            "/configurator/set_overlays",
            data={"show_grid": "false", "show_objects": "true"},
        )
        assert r.status_code == 200
        assert configurator_app.state.cc.show_grid is False
        assert configurator_app.state.cc.show_objects is True

    def test_grid_config_updates_hex(self, configurator_client, configurator_app):
        r = configurator_client.post(
            "/configurator/grid_configuration", data={"size": "30"}
        )
        assert r.status_code == 200
        assert configurator_app.state.cc.hex.size == 30.0
