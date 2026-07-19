document.addEventListener('DOMContentLoaded', function() {
    // Global Event Delegation for all previously inline handlers
    document.addEventListener('click', function(e) {
        if (!e.target || typeof e.target.closest !== 'function') return;
        if (e.target.closest('#tab-harmony')) switchTab('harmony');
        if (e.target.closest('#tab-compcon')) switchTab('compcon');
        if (e.target.closest('.clearSelectionBtn')) clearSelection();
        if (e.target.closest('#editSessionToggleBtn')) {
            let form = document.getElementById('updateSessionForm');
            if (form) form.hidden = !form.hidden;
        }
        if (e.target.closest('#renameModalSubmit')) doRename();
        if (e.target.closest('.chat-nav-btn')) {
            window.currentChatChannel = e.target.closest('.chat-nav-btn').getAttribute('data-channel');
        }
        if (e.target.closest('#sendChatBtn') || e.target.closest('#chatSendBtn')) sendChatMessage();
        if (e.target.closest('#chatToggleButton') || e.target.closest('.closeChatBtn')) {
            if (typeof toggleChatVisibility === 'function') toggleChatVisibility();
        }
        if (e.target.closest('#expandViewerBtn')) {
            if (typeof toggleGameWorldFullscreen === 'function') toggleGameWorldFullscreen();
        }
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
    let viewIdInput = document.getElementById('viewId');
    let vId = viewIdInput ? viewIdInput.value : '';
    window.chatWs = new WebSocket(chatWsProtocol + '//' + window.location.host + '/harmony/chat_ws?view_id=' + encodeURIComponent(vId));
    
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
            
            let chatWidget = document.getElementById('discordChatWidget');
            let toggleBtn = document.getElementById('chatToggleButton');
            if (chatWidget && chatWidget.style.display === 'none' && toggleBtn) {
                toggleBtn.innerHTML = '💬 Open Chat (New!)';
                toggleBtn.classList.remove('btn-outline-primary');
                toggleBtn.classList.add('btn-warning');
            }
        }
    };
    
    // Sync chat tabs periodically
    setInterval(syncChatTabs, 1000);



    // Collapse persistence logic
    if (!window._harmonyCollapseBound) {
        window._harmonyCollapseBound = true;
        document.body.addEventListener('hide.bs.collapse', function(e) {
            if (e.target.id && e.target.id.startsWith('collapse-')) {
                sessionStorage.setItem('collapseState_' + e.target.id, 'hidden');
            }
        });
        document.body.addEventListener('show.bs.collapse', function(e) {
            if (e.target.id && e.target.id.startsWith('collapse-')) {
                sessionStorage.setItem('collapseState_' + e.target.id, 'shown');
            }
        });
        document.body.addEventListener('htmx:afterSwap', function(e) {
            if (e.target.id === 'objectFilterRetriever') {
                document.querySelectorAll('.object-group-collapse').forEach(function(el) {
                    var state = sessionStorage.getItem('collapseState_' + el.id);
                    if (state === 'shown') {
                        el.classList.add('show');
                    } else {
                        el.classList.remove('show');
                    }
                });
            }
        });
    }
});

// Ensure canvas initializes after all scripts and images are fully loaded
window.addEventListener('load', function() {
    if (typeof initHarmonyCanvas === 'function') {
        initHarmonyCanvas();
        let viewIdInput = document.getElementById('viewId');
        if (viewIdInput) {
            let viewId = viewIdInput.value;
            syncCanvasData(viewId);
            let sse = new EventSource('/harmony/canvas_stream/' + viewId);
            sse.onmessage = function(event) {
                if (event.data === "update") {
                    syncCanvasData(viewId);
                }
            };
        }
    }
});

window.syncChatTabs = function() {
    let cd = window.harmonyCanvasData;
    if (!cd || !cd.ally_groups) return;
    
    let tabsUl = document.getElementById('chatTabs');
    let contentDiv = document.getElementById('chatTabContent');
    if (!tabsUl || !contentDiv) return;

    let neededChannels = ['group'];
    cd.ally_groups.forEach(g => neededChannels.push(g));
    // Remove "dms" and "team" if they exist, let's keep only what's in neededChannels
    
    let currentChannels = Array.from(tabsUl.querySelectorAll('.chat-nav-btn')).map(b => b.getAttribute('data-channel'));
    
    // Check if we need to rebuild
    let needsRebuild = false;
    if (neededChannels.length !== currentChannels.length) needsRebuild = true;
    else {
        for(let i=0; i<neededChannels.length; i++) {
            if(neededChannels[i] !== currentChannels[i]) needsRebuild = true;
        }
    }
    
    if (needsRebuild) {
        tabsUl.innerHTML = '';
        
        let existingContents = new Map();
        Array.from(contentDiv.children).forEach(child => {
            let ch = child.getAttribute('data-channel-content');
            if (ch) existingContents.set(ch, child);
        });
        
        contentDiv.innerHTML = '';
        
        neededChannels.forEach((ch, idx) => {
            let li = document.createElement('li');
            li.className = 'nav-item';
            li.setAttribute('role', 'presentation');
            
            let btn = document.createElement('button');
            let isActive = window.currentChatChannel === ch || (idx === 0 && !neededChannels.includes(window.currentChatChannel));
            if (isActive) window.currentChatChannel = ch;
            
            btn.className = 'nav-link chat-nav-btn' + (isActive ? ' active' : '');
            btn.setAttribute('data-bs-toggle', 'tab');
            btn.setAttribute('data-bs-target', '#chat-' + ch);
            btn.setAttribute('type', 'button');
            btn.setAttribute('role', 'tab');
            btn.setAttribute('data-channel', ch);
            btn.innerText = ch.charAt(0).toUpperCase() + ch.slice(1);
            li.appendChild(btn);
            tabsUl.appendChild(li);
            
            let pane = existingContents.get(ch);
            if (!pane) {
                pane = document.createElement('div');
                pane.className = 'tab-pane fade h-100';
                pane.id = 'chat-' + ch;
                pane.setAttribute('role', 'tabpanel');
                pane.setAttribute('data-channel-content', ch);
                pane.innerHTML = `<div id="chat-messages-${ch}" style="height: 100%; overflow-y: auto;"></div>`;
            }
            if (isActive) {
                pane.classList.add('show', 'active');
            } else {
                pane.classList.remove('show', 'active');
            }
            contentDiv.appendChild(pane);
        });
    }
}

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

window.toggleGameWorldFullscreen = function() {
    const wrapper = document.getElementById('GameWorldWrapper');
    const btn = document.getElementById('expandViewerBtn');
    if (!wrapper || !btn) return;
    
    wrapper.classList.toggle('fullscreen-viewer');
    document.body.classList.toggle('fullscreen-active');
    
    if (wrapper.classList.contains('fullscreen-viewer')) {
        btn.innerText = '⤡ Collapse';
        btn.classList.replace('btn-outline-info', 'btn-outline-warning');
    } else {
        btn.innerText = '⤢ Expand';
        btn.classList.replace('btn-outline-warning', 'btn-outline-info');
    }
    
    // Trigger a resize event to ensure Canvas redraws
    setTimeout(() => window.dispatchEvent(new Event('resize')), 100);
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
        if (toggleBtn) {
            toggleBtn.innerHTML = '💬 Close Chat';
            toggleBtn.classList.remove('btn-warning');
            toggleBtn.classList.add('btn-outline-primary');
        }
    } else {
        chatWidget.style.display = 'none';
        if (toggleBtn) toggleBtn.innerHTML = '💬 Open Chat';
    }
}

window.sendChatMessage = function() {
    if (window.chatStatus !== 'Running') return;
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

window.onChatStatusUpdate = function(status) {
    let indicator = document.getElementById('chatStatusIndicator');
    if (indicator) indicator.textContent = status;
    
    let widgetStatus = document.getElementById('widgetChatStatus');
    if (widgetStatus) widgetStatus.textContent = '(' + status + ')';
    
    let input = document.getElementById('chatInput');
    let btn = document.getElementById('sendChatBtn') || document.getElementById('chatSendBtn');
    
    if (status !== 'Running') {
        if (input) {
            input.disabled = true;
            input.placeholder = "Chat is " + status;
        }
        if (btn) btn.disabled = true;
    } else {
        if (input) {
            input.disabled = false;
            input.placeholder = "Type a message...";
        }
        if (btn) btn.disabled = false;
    }
};
