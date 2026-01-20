
import pytest
import unittest.mock as mock
import json
from harmony.harmonyServer import SESSIONS, SessionConfig

# Mock generators
def mock_gen(*args, **kwargs):
    yield b'--frame\r\nContent-Type: image/jpg\r\n\r\nfake\r\n'

@pytest.fixture
def harmony_client(harmony_app):
    return harmony_app.test_client()

def test_select_pixel_additional_cells(harmony_client):
    """Test selecting multiple pixels adds additional info to interactor template."""
    app = harmony_client.application
    app.cc.camCoordToAxial.return_value = (1, 2)
    
    # Mocking dependencies
    app.cm.cc.axial_distance.return_value = 5
    app.cm.memory = [] # No overlaps for this test

    view_id = 'view1'
    SESSIONS[view_id] = SessionConfig()

    # Select first pixel
    data1 = {'viewId': view_id, 'selectedPixel': '[100, 200]', 'selectedCamera': 'Camera 0', 'appendPixel': ''}
    response1 = harmony_client.post('/harmony/select_pixel', data=data1)
    assert response1.status_code == 200
    
    # Create manual selection state simulating multiple pixels
    # Since select_pixel logic for appending depends on logic we want to test or we can just mock the state
    # But let's try to use the endpoint if possible. 
    # The endpoint calls: current_app.cm.cc.camCoordToAxial(cam, (x, y)) -> returns (1,2) mocked
    
    # Let's mock camCoordToAxial to return different values
    app.cc.camCoordToAxial.side_effect = [(0, 0), (0, 1), (0, 2)] 
    
    # 1. Select First Cell
    data1 = {'viewId': view_id, 'selectedPixel': '[0, 0]', 'selectedCamera': 'Camera 0', 'appendPixel': ''}
    harmony_client.post('/harmony/select_pixel', data=data1)
    
    # 2. Select Second Cell (Append)
    # The frontend usually sends appendPixel=true if shift/ctrl click.
    # Logic in server: if existing.firstCell and existing.secondCell is None: ... additionalCells = [axial]
    
    data2 = {'viewId': view_id, 'selectedPixel': '[0, 1]', 'selectedCamera': 'Camera 0', 'appendPixel': 'true'}
    response2 = harmony_client.post('/harmony/select_pixel', data=data2)
    
    assert SESSIONS[view_id].selection.firstCell == (0, 0)
    assert len(SESSIONS[view_id].selection.additionalCells) == 1
    assert SESSIONS[view_id].selection.additionalCells[0] == (0, 1)

    # 3. Select Third Cell (Append)
    data3 = {'viewId': view_id, 'selectedPixel': '[0, 2]', 'selectedCamera': 'Camera 0', 'appendPixel': 'true'}
    response3 = harmony_client.post('/harmony/select_pixel', data=data3)
    
    assert len(SESSIONS[view_id].selection.additionalCells) == 2
    
    response_text = response3.data.decode()
    print(response_text)
    
    # Verify "Additional cells" text is present
    assert "Additional cells: 2" in response_text
    # Verify First Cell text
    assert "Selected cell: (0, 0)" in response_text
    # Verify Distance (to second cell - which is additionalCells[0] = (0,1))
    # Distance from (0,0) to (0,1) is 1. Mock returned 5 earlier, let's reset or rely on side_effect if used.
    # Actually app.cm.cc.axial_distance was mocked to return 5.
    assert "Distance to (0, 1)  --  5 cells" in response_text 

