import pytest
import json
import os
from observer.HexObserver import HexCaptureConfiguration, HexGridConfiguration

def test_grid_export_and_reload(tmp_path):
    # Create an initial configuration
    cc = HexCaptureConfiguration()
    cc.hex = HexGridConfiguration(size=35.0, offset_xy=(5.0, 5.0), rotation_deg=15.0)
    cc.show_grid = True
    cc.show_objects = True

    # Build the configuration dict
    config_dict = cc.buildConfiguration()

    # Verify grid info is present in the exported config
    assert "hex" in config_dict
    assert config_dict["hex"]["size"] == 35.0
    assert config_dict["show_grid"] is True

    # Save to a temporary file
    config_path = tmp_path / "observerConfiguration.json"
    with open(config_path, "w") as f:
        json.dump(config_dict, f)

    # Create a new instance and load from the file
    cc2 = HexCaptureConfiguration()
    cc2.loadConfiguration(path=str(config_path))

    # Verify grid info is correctly restored
    assert cc2.hex is not None
    assert cc2.hex.size == 35.0
    assert list(cc2.hex.offset_xy) == list(cc.hex.offset_xy)
    assert cc2.hex.rotation_deg == 15.0
    assert getattr(cc2, "show_grid", False) is True
    assert getattr(cc2, "show_objects", False) is True

