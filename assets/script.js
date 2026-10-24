// Sidebar toggle for mobile
document.addEventListener('DOMContentLoaded', function() {
    const sidebar = document.getElementById('sidebar');
    const sidebarToggle = document.getElementById('sidebar-toggle');

    function setSidebarOpen(open) {
        sidebar.classList.toggle('active', open);
        sidebarToggle.setAttribute('aria-expanded', String(open));
        sidebarToggle.setAttribute('aria-label', open ? '关闭目录' : '打开目录');
    }

    if (sidebarToggle) {
        sidebarToggle.addEventListener('click', function() {
            setSidebarOpen(!sidebar.classList.contains('active'));
        });
    }

    // Close sidebar when clicking outside on mobile
    document.addEventListener('click', function(event) {
        const isClickInside = sidebar.contains(event.target);
        const isToggleClick = sidebarToggle.contains(event.target);

        if (!isClickInside && !isToggleClick && sidebar.classList.contains('active')) {
            setSidebarOpen(false);
        }
    });

    // Smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
                if (window.innerWidth <= 768) {
                    setSidebarOpen(false);
                }
            }
        });
    });

    // Pure CSS Tree Navigation
    const treeNav = document.querySelector('.tree-nav');
    if (treeNav) {
        const TREE_STATE_KEY = 'tree-nav-state';

        // Load saved state
        function loadTreeState() {
            try {
                const saved = localStorage.getItem(TREE_STATE_KEY);
                return saved ? JSON.parse(saved) : {};
            } catch (e) {
                return {};
            }
        }

        // Save tree state
        function saveTreeState() {
            const state = {};
            document.querySelectorAll('.tree-folder').forEach(folder => {
                const header = folder.querySelector('.tree-folder-header');
                if (header) {
                    const title = header.querySelector('.tree-title');
                    if (title) {
                        const folderPath = title.textContent.trim();
                        state[folderPath] = folder.classList.contains('expanded');
                    }
                }
            });
            try {
                localStorage.setItem(TREE_STATE_KEY, JSON.stringify(state));
            } catch (e) {
                // Ignore localStorage errors
            }
        }

        // Apply saved state
        function applyTreeState() {
            const state = loadTreeState();
            document.querySelectorAll('.tree-folder').forEach(folder => {
                const header = folder.querySelector('.tree-folder-header');
                if (header) {
                    const title = header.querySelector('.tree-title');
                    if (title) {
                        const folderPath = title.textContent.trim();
                        // Check if folder contains active item
                        const hasActive = folder.querySelector('.tree-item.active');

                        if (hasActive) {
                            // Always expand folders containing active items
                            folder.classList.add('expanded');
                        } else if (state[folderPath] !== undefined) {
                            // Apply saved state for other folders
                            if (state[folderPath]) {
                                folder.classList.add('expanded');
                            } else {
                                folder.classList.remove('expanded');
                            }
                        }
                    }
                }
            });
        }

        // Initialize state on page load
        setTimeout(() => {
            applyTreeState();
        }, 0);

        // Handle folder clicks
        document.querySelectorAll('.tree-folder-header').forEach(header => {
            header.addEventListener('click', function(e) {
                e.preventDefault();
                const folder = this.parentElement;
                folder.classList.toggle('expanded');
                saveTreeState();
            });
        });

        // Search functionality
        const searchInput = document.getElementById('sidebar-search-input');
        const searchClear = document.getElementById('sidebar-search-clear');
        if (searchInput) {
            let searchTimeout;

            function setClearVisible(visible) {
                if (searchClear) {
                    searchClear.style.display = visible ? 'block' : 'none';
                }
            }

            function setNoResultsVisible(visible) {
                let noResultsMsg = document.getElementById('no-search-results');
                if (visible) {
                    if (!noResultsMsg) {
                        noResultsMsg = document.createElement('div');
                        noResultsMsg.id = 'no-search-results';
                        noResultsMsg.className = 'no-results';
                        noResultsMsg.textContent = '没有找到匹配的结果';
                        treeNav.parentNode.insertBefore(noResultsMsg, treeNav);
                    }
                    noResultsMsg.style.display = 'block';
                } else if (noResultsMsg) {
                    noResultsMsg.style.display = 'none';
                }
            }

            function performTreeSearch() {
                const query = searchInput.value.toLowerCase().trim();
                setClearVisible(Boolean(query));
                const items = document.querySelectorAll('.tree-item');
                const folders = document.querySelectorAll('.tree-folder');

                // Clear previous matches
                document.querySelectorAll('.search-match').forEach(el => {
                    el.classList.remove('search-match');
                });

                if (!query) {
                    // Show all items
                    items.forEach(item => item.style.display = '');
                    folders.forEach(folder => folder.style.display = '');
                    setNoResultsVisible(false);
                    applyTreeState();
                    return;
                }

                // Search and highlight
                let hasResults = false;
                items.forEach(item => {
                    const titleEl = item.querySelector('.tree-title');
                    const title = titleEl ? titleEl.textContent.toLowerCase() : '';
                    if (title.includes(query)) {
                        item.style.display = '';
                        item.classList.add('search-match');
                        hasResults = true;

                        // Expand parent folders
                        let parent = item.parentElement;
                        while (parent && parent !== treeNav) {
                            if (parent.classList.contains('tree-folder')) {
                                parent.classList.add('expanded');
                                parent.style.display = '';
                            }
                            parent = parent.parentElement;
                        }
                    } else {
                        item.style.display = 'none';
                    }
                });

                // Hide empty folders
                folders.forEach(folder => {
                    const visibleItems = folder.querySelectorAll('.tree-item:not([style*="none"])');
                    if (visibleItems.length === 0) {
                        folder.style.display = 'none';
                    } else {
                        folder.style.display = '';
                    }
                });

                setNoResultsVisible(!hasResults);
            }

            function scheduleTreeSearch() {
                clearTimeout(searchTimeout);
                searchTimeout = setTimeout(performTreeSearch, 250);
            }

            searchInput.addEventListener('input', scheduleTreeSearch);

            searchInput.addEventListener('keydown', function(e) {
                if (e.key === 'Escape') {
                    this.value = '';
                    performTreeSearch();
                }
            });

            if (searchClear) {
                searchClear.addEventListener('click', function() {
                    searchInput.value = '';
                    performTreeSearch();
                    searchInput.focus();
                });

                searchClear.addEventListener('keydown', function(e) {
                    if (e.key === 'Enter' || e.key === ' ') {
                        e.preventDefault();
                        this.click();
                    }
                });
            }
        }
    }

    // Flat navigation search (for small doc sets)
    const navList = document.querySelector('.nav-list');
    if (navList) {
        const searchInput = document.getElementById('sidebar-search-input');
        const searchClear = document.getElementById('sidebar-search-clear');

        if (searchInput) {
            let searchTimeout;

            function setClearVisible(visible) {
                if (searchClear) {
                    searchClear.style.display = visible ? 'block' : 'none';
                }
            }

            function setNoResultsVisible(visible) {
                let noResultsMsg = document.getElementById('no-search-results');
                if (visible) {
                    if (!noResultsMsg) {
                        noResultsMsg = document.createElement('div');
                        noResultsMsg.id = 'no-search-results';
                        noResultsMsg.className = 'no-results';
                        noResultsMsg.textContent = '没有找到匹配的结果';
                        navList.parentNode.insertBefore(noResultsMsg, navList);
                    }
                    noResultsMsg.style.display = 'block';
                } else if (noResultsMsg) {
                    noResultsMsg.style.display = 'none';
                }
            }

            function performFlatSearch() {
                const query = searchInput.value.toLowerCase().trim();
                setClearVisible(Boolean(query));

                let visibleCount = 0;
                navList.querySelectorAll('li').forEach(item => {
                    const link = item.querySelector('a');
                    const text = link ? link.textContent.toLowerCase() : '';

                    if (!query || text.includes(query)) {
                        item.style.display = '';
                        visibleCount++;
                    } else {
                        item.style.display = 'none';
                    }
                });

                setNoResultsVisible(Boolean(query) && visibleCount === 0);
            }

            function scheduleFlatSearch() {
                clearTimeout(searchTimeout);
                searchTimeout = setTimeout(performFlatSearch, 150);
            }

            searchInput.addEventListener('input', scheduleFlatSearch);

            searchInput.addEventListener('keydown', function(e) {
                if (e.key === 'Escape') {
                    this.value = '';
                    performFlatSearch();
                }
            });

            if (searchClear) {
                searchClear.addEventListener('click', function() {
                    searchInput.value = '';
                    performFlatSearch();
                    searchInput.focus();
                });

                searchClear.addEventListener('keydown', function(e) {
                    if (e.key === 'Enter' || e.key === ' ') {
                        e.preventDefault();
                        this.click();
                    }
                });
            }

            performFlatSearch();
        }
    }
});
