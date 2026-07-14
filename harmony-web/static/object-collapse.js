if (!window.objectCollapseStateTracked) {
    window.expandedObjectGroups = new Set();
    document.addEventListener('show.bs.collapse', function (e) {
        if (e.target && e.target.classList && e.target.classList.contains('object-group-collapse')) {
            window.expandedObjectGroups.add(e.target.id);
        }
    });
    document.addEventListener('hide.bs.collapse', function (e) {
        if (e.target && e.target.classList && e.target.classList.contains('object-group-collapse')) {
            window.expandedObjectGroups.delete(e.target.id);
        }
    });

    document.body.addEventListener('htmx:beforeSwap', function(evt) {
        if ((evt.detail.target.id === 'objectFilterRetriever' || evt.detail.target.id === 'objectTable') && window.expandedObjectGroups) {
            let html = evt.detail.serverResponse;
            window.expandedObjectGroups.forEach(id => {
                let search = `class="collapse object-group-collapse" id="${id}"`;
                let replace = `class="collapse object-group-collapse show" id="${id}"`;
                html = html.split(search).join(replace);
                
                let btnRegex = new RegExp(`data-bs-target="#${id}"\\s*aria-expanded="false"`, 'g');
                html = html.replace(btnRegex, `data-bs-target="#${id}" aria-expanded="true"`);
            });
            evt.detail.serverResponse = html;
        }
    });

    window.objectCollapseStateTracked = true;
}
