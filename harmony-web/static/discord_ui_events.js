document.addEventListener('DOMContentLoaded', function() {
    // Global Event Delegation for all previously inline handlers
    document.addEventListener('click', function(e) {
        if (e.target.closest('.cam-btn')) {
            gameWorldClick(e.target.closest('.cam-btn').getAttribute('data-camera'));
        }
        if (e.target.closest('#chatToggleButton') || e.target.closest('.closeChatBtn')) {
            toggleChatVisibility();
        }
        if (e.target.closest('#tab-harmony')) switchTab('harmony');
        if (e.target.closest('#tab-compcon')) switchTab('compcon');
        if (e.target.closest('.clearSelectionBtn')) clearSelection();
        if (e.target.closest('#editSessionToggleBtn')) {
            let form = document.getElementById('updateSessionForm');
            if (form) form.hidden = !form.hidden;
        }
        if (e.target.closest('#renameModalSubmit')) doRename();
        if (e.target.closest('#group-tab')) window.currentChatChannel = 'group';
        if (e.target.closest('#team-tab')) window.currentChatChannel = 'team';
        if (e.target.closest('#dms-tab')) window.currentChatChannel = 'dms';
        if (e.target.closest('#sendChatBtn')) sendChatMessage();
    });

    document.addEventListener('change', function(e) {
        if (e.target.id === 'layoutSplit') toggleLayout('split');
        if (e.target.id === 'layoutTabs') toggleLayout('tabs');
        if (e.target.id === 'showGridHarmony' || e.target.id === 'showObjectsHarmony') toggleHarmonyOverlays();
        
        if (e.target.name === 'toolMode') setToolMode(e.target.value);
        if (e.target.id === 'burstRadius' || e.target.id === 'coneLength') updateToolParams();
    });

    document.addEventListener('submit', function(e) {
        if (e.target.id === 'createObjectForm') {
            e.preventDefault();
            submitCreateObject(e);
        }
    });

    document.addEventListener('keypress', function(e) {
        if (e.target.id === 'chatInput' && e.key === 'Enter') {
            sendChatMessage();
        }
        if (e.target.id === 'renameModalInput' && e.key === 'Enter') {
            doRename();
        }
    });
    
    // Initialize Chat WebSocket
    window.currentChatChannel = 'group';
    let chatWsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    window.chatWs = new WebSocket(chatWsProtocol + '//' + window.location.host + '/harmony/chat_ws');
    
    window.chatWs.onmessage = function(event) {
        let msg = JSON.parse(event.data);
        let channel = msg.channel || 'group';
        let container = document.getElementById('chat-messages-' + channel);
        if (container) {
            let msgEl = document.createElement('div');
            msgEl.style.marginBottom = '8px';
            msgEl.style.fontSize = '0.9rem';
            msgEl.innerHTML = '<strong style="color: #5865F2;">' + msg.author + ':</strong> ' + msg.content;
            container.appendChild(msgEl);
            container.scrollTop = container.scrollHeight;
        }
    };
});

window.toggleHarmonyOverlays = function() {
    let formData = new URLSearchParams();
    let showGridElem = document.getElementById('showGridHarmony');
    let showObjectsElem = document.getElementById('showObjectsHarmony');
    let viewIdElem = document.getElementById('viewId');
    if (!showGridElem || !showObjectsElem || !viewIdElem) return;
    
    formData.append('show_grid', showGridElem.checked);
    formData.append('show_objects', showObjectsElem.checked);
    formData.append('viewId', viewIdElem.value);
    fetch('/harmony/set_overlays', { method: 'POST', body: formData });
    
    if (window.harmonyEditor) window.harmonyEditor.render();
    if (window.harmonyEditors) window.harmonyEditors.forEach(ed => ed.render());
}

window.submitCreateObject = function(event) {
    let cells = [];
    if (window.toolMode && window.toolMode !== 'none' && window.currentToolHighlight && window.currentToolHighlight.length > 0) {
        cells = window.currentToolHighlight;
    } else {
        let sel = window.harmonyCanvasData ? window.harmonyCanvasData.selection : null;
        if (sel && sel.firstCell) {
            cells.push([sel.firstCell._q, sel.firstCell._r]);
            if (sel.additionalCells) {
                sel.additionalCells.forEach(c => cells.push([c._q, c._r]));
            }
        }
    }
    
    if (cells.length === 0) {
        alert("No cells selected!");
        return;
    }
    
    let name = document.getElementById('createObjectName').value;
    let formData = new URLSearchParams();
    formData.append('name', name);
    formData.append('cells', JSON.stringify(cells));
    
    fetch('/harmony/objects', {
        method: 'POST',
        body: formData,
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' }
    }).then(r => {
        document.getElementById('createObjectName').value = '';
        if (typeof clearSelection === 'function') clearSelection();
    });
}

window.toggleLayout = function(mode) {
    const harmonyCol = document.getElementById('harmonyColumn');
    const compconCol = document.getElementById('compconColumn');
    const navTabs = document.getElementById('layoutNavTabs');
    if (!harmonyCol || !compconCol || !navTabs) return;
    
    if (mode === 'split') {
        navTabs.style.display = 'none';
        harmonyCol.className = 'col-5';
        harmonyCol.style.display = 'block';
        compconCol.className = 'col-7';
        compconCol.style.display = 'block';
    } else {
        navTabs.style.display = 'flex';
        harmonyCol.className = 'col-12';
        compconCol.className = 'col-12';
        switchTab('harmony');
    }
    if (window.harmonyEditor) window.harmonyEditor.render();
    if (window.harmonyEditors) window.harmonyEditors.forEach(ed => ed.render());
}

window.switchTab = function(tab) {
    const harmonyCol = document.getElementById('harmonyColumn');
    const compconCol = document.getElementById('compconColumn');
    const tabHarmony = document.getElementById('tab-harmony');
    const tabCompcon = document.getElementById('tab-compcon');
    const layoutTabs = document.getElementById('layoutTabs');
    if (!harmonyCol || !compconCol || !tabHarmony || !tabCompcon || !layoutTabs) return;
    
    if (layoutTabs.checked) {
        if (tab === 'harmony') {
            harmonyCol.style.display = 'block';
            compconCol.style.display = 'none';
            tabHarmony.classList.add('active');
            tabCompcon.classList.remove('active');
        } else {
            harmonyCol.style.display = 'none';
            compconCol.style.display = 'block';
            tabHarmony.classList.remove('active');
            tabCompcon.classList.add('active');
        }
    }
}

window.toggleChatVisibility = function() {
    const chatWidget = document.getElementById('discordChatWidget');
    const toggleBtn = document.getElementById('chatToggleButton');
    if (!chatWidget) return;
    if (chatWidget.style.display === 'none') {
        chatWidget.style.display = 'flex';
        if (toggleBtn) toggleBtn.innerHTML = '💬 Close Chat';
    } else {
        chatWidget.style.display = 'none';
        if (toggleBtn) toggleBtn.innerHTML = '💬 Open Chat';
    }
}

window.sendChatMessage = function() {
    let input = document.getElementById('chatInput');
    if (!input) return;
    let text = input.value.trim();
    if (text !== '') {
        let msg = {
            author: document.getElementById('viewId').value,
            content: text,
            channel: window.currentChatChannel,
            timestamp: 0,
            from_discord: false
        };
        if (window.chatWs && window.chatWs.readyState === WebSocket.OPEN) {
            window.chatWs.send(JSON.stringify(msg));
        }
        input.value = '';
    }
}

var _oid = null;
var _modal = null;

window.openRenameModal = function(oid) {
    _oid = oid;
    document.getElementById('renameModalCurrentName').textContent = oid;
    var input = document.getElementById('renameModalInput');
    input.value = oid;
    document.getElementById('renameModalError').hidden = true;
    if (!_modal) {
        _modal = new bootstrap.Modal(document.getElementById('renameUnitModal'));
    }
    _modal.show();
    document.getElementById('renameUnitModal').addEventListener('shown.bs.modal', function () {
        input.focus();
        input.select();
    }, { once: true });
};

window.doRename = function() {
    var newName = document.getElementById('renameModalInput').value.trim();
    var errEl = document.getElementById('renameModalError');
    if (!newName || !_oid) return;
    var fd = new FormData();
    fd.append('objectName', newName);
    fetch('/harmony/objects/' + encodeURIComponent(_oid), { method: 'POST', body: fd })
        .then(function (resp) {
            if (resp.ok) {
                _modal.hide();
                if (window.htmx) {
                    window.htmx.ajax('GET', '/harmony/objects?isUserUI=true&viewId=' + document.getElementById('viewId').value,
                        { target: '#objectFilterRetriever' });
                }
            } else {
                resp.text().then(function (t) {
                    errEl.textContent = t;
                    errEl.hidden = false;
                });
            }
        });
}
