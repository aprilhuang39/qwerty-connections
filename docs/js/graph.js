// Graph configuration
const config = {
    width: null,  // Will be set dynamically
    height: null, // Will be set dynamically
    nodeRadiusRange: [5, 30],
    linkStrengthRange: [0.1, 1],
    nodeStrength: -100,
    linkDistance: 150,
    charge: -190,
    edgeFilterPercentage: 10, // Default to show only 10% of edges
    colors: {
        hasPD: 'red',
        noPD: 'blue'
    }
};

// State management
let state = {
    nodes: [],
    links: [],
    allLinks: [], // Store all links before filtering
    simulation: null,
    svg: null,
    edgeFile: null, // Will be set dynamically from available files
    loading: true
};

// DOM References
const graphContainer = document.getElementById('graph-container');
const svg = d3.select('#graph');
const edgeSelect = document.getElementById('edge-select');
const chargeSlider = document.getElementById('charge-slider');
const chargeValue = document.getElementById('charge-value');
const linkDistanceSlider = document.getElementById('link-distance-slider');
const linkDistanceValue = document.getElementById('link-distance-value');
const edgeFilterSlider = document.getElementById('edge-filter-slider');
const edgeFilterValue = document.getElementById('edge-filter-value');
const loadingElement = document.getElementById('loading');

// Initialize graph dimensions
function initGraphDimensions() {
    config.width = graphContainer.clientWidth;
    config.height = graphContainer.clientHeight;

    svg.attr('width', config.width)
       .attr('height', config.height);
}

// Scale functions
function getNodeRadius(updrsScore) {
    // Create a scale for node radius based on UPDRS-III score
    // Min radius for score 0, max radius for highest score
    const scoreExtent = d3.extent(state.nodes, d => d['UPDRS-III']);
    const radiusScale = d3.scaleLinear()
        .domain([0, scoreExtent[1]])
        .range(config.nodeRadiusRange);
    
    return radiusScale(updrsScore);
}

function getLinkStrength(distance) {
    // Create a scale for link strength based on distance
    // Higher distance = weaker link
    const distanceExtent = d3.extent(state.links, d => d.distance);
    const strengthScale = d3.scaleLinear()
        .domain(distanceExtent)
        .range(config.linkStrengthRange);
    
    // Invert so higher distance = weaker link
    return 1 - strengthScale(distance);
}

// Loading state management
function setLoading(isLoading) {
    state.loading = isLoading;
    loadingElement.style.display = isLoading ? 'block' : 'none';
}

// Load data
async function loadData() {
    try {
        setLoading(true);
        
        // Load available edge files and populate dropdown
        await loadEdgeFileOptions();
        
        // Load nodes data
        const nodesResponse = await fetch('force_graph/nodes.json');
        if (!nodesResponse.ok) throw new Error('Failed to load nodes data');
        state.nodes = await nodesResponse.json();
        
        // Load links data based on selected edge file
        await loadLinks(state.edgeFile);

        // Initialize the force graph
        initForceGraph();
        
        setLoading(false);
    } catch (error) {
        console.error('Error loading data:', error);
        alert('Failed to load graph data. Please try again later.');
        setLoading(false);
    }
}

async function loadLinks(edgeFile) {
    try {
        setLoading(true);
        
        const linksResponse = await fetch(`force_graph/${edgeFile}`);
        if (!linksResponse.ok) throw new Error('Failed to load links data');
        
        const rawLinks = await linksResponse.json();
        
        // Process the links to use the proper format for D3 force graph
        state.allLinks = rawLinks.map(link => ({
            source: link.pID_1,
            target: link.pID_2,
            distance: link.distance
        }));
        
        // Apply filtering
        filterLinks();
        
        setLoading(false);
    } catch (error) {
        console.error('Error loading links:', error);
        alert('Failed to load edge data. Please try again later.');
        setLoading(false);
    }
}

// Filter links based on edge filter percentage
function filterLinks() {
    if (!state.allLinks || state.allLinks.length === 0) return;
    
    // Sort links by distance (smaller distances = stronger connections)
    const sortedLinks = [...state.allLinks].sort((a, b) => a.distance - b.distance);
    
    // Calculate how many links to keep based on percentage
    const keepCount = Math.ceil(sortedLinks.length * (config.edgeFilterPercentage / 100));
    
    // Get the n% of links with smallest distances
    state.links = sortedLinks.slice(0, keepCount);
}

// Initialize force graph
function initForceGraph() {
    // Clear previous graph
    svg.selectAll('*').remove();
    
    // Add a group for zoom behavior
    const g = svg.append('g').attr('class', 'graph-container');
    
    // Create simulation
    state.simulation = d3.forceSimulation(state.nodes)
        .force('link', d3.forceLink(state.links)
            .id(d => d.pID)
            .distance(config.linkDistance)
            .strength(d => getLinkStrength(d.distance))
        )
        .force('charge', d3.forceManyBody().strength(config.charge))
        .force('center', d3.forceCenter(config.width / 2, config.height / 2))
        .force('collision', d3.forceCollide().radius(d => getNodeRadius(d['UPDRS-III']) + 1));

    // Create links
    const link = g.append('g')
        .attr('class', 'links')
        .selectAll('line')
        .data(state.links)
        .enter()
        .append('line')
        .attr('class', 'link')
        .style('stroke-width', d => Math.max(1, 5 * (1 - getLinkStrength(d.distance))));
    
    // Create nodes
    const node = g.append('g')
        .attr('class', 'nodes')
        .selectAll('circle')
        .data(state.nodes)
        .enter()
        .append('circle')
        .attr('class', 'node')
        .attr('r', d => getNodeRadius(d['UPDRS-III']))
        .style('fill', d => d.has_PD ? config.colors.hasPD : config.colors.noPD)
        .call(d3.drag()
            .on('start', dragstarted)
            .on('drag', dragged)
            .on('end', dragended)
        );
    
    // Add tooltips
    node.append('title')
        .text(d => `pID: ${d.pID}\nhas_PD: ${d.has_PD}\nUPDRS-III: ${d['UPDRS-III']}`);
    
    // Add node labels
    const nodeLabels = g.append('g')
        .attr('class', 'node-labels')
        .selectAll('text')
        .data(state.nodes)
        .enter()
        .append('text')
        .attr('class', 'node-label')
        .text(d => d.pID)
        .style('display', (d) => getNodeRadius(d['UPDRS-III']) > 10 ? 'block' : 'none');
    
    // Update simulation on tick
    state.simulation.on('tick', () => {
        link
            .attr('x1', d => d.source.x)
            .attr('y1', d => d.source.y)
            .attr('x2', d => d.target.x)
            .attr('y2', d => d.target.y);
        
        node
            .attr('cx', d => d.x)
            .attr('cy', d => d.y);
            
        nodeLabels
            .attr('x', d => d.x)
            .attr('y', d => d.y + 3);
    });
    
    // Zoom functionality
    const zoom = d3.zoom()
        .scaleExtent([0.1, 10])
        .on('zoom', (event) => {
            g.attr('transform', event.transform);
            
            // Show/hide labels based on zoom level
            const currentScale = event.transform.k;
            nodeLabels.style('display', (d) => 
                (getNodeRadius(d['UPDRS-III']) * currentScale > 10) ? 'block' : 'none'
            );
        });
    
    svg.call(zoom);
    
    // Fit all nodes in view after simulation stabilizes
    state.simulation.on('end', () => {
        fitGraphToView(g, zoom);
    });
    
    // Also attempt to fit after a short delay in case simulation takes a long time
    setTimeout(() => {
        fitGraphToView(g, zoom);
    }, 2000);
}

// Fit the graph to the view so all nodes are visible
function fitGraphToView(g, zoom) {
    if (!state.nodes.length) return;
    
    // Get current bounds of all nodes
    const nodeElements = g.selectAll('.node');
    if (nodeElements.empty()) return;
    
    // Calculate the bounding box
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    
    nodeElements.each(function() {
        const node = d3.select(this);
        const cx = parseFloat(node.attr('cx')) || 0;
        const cy = parseFloat(node.attr('cy')) || 0;
        const r = parseFloat(node.attr('r')) || 0;
        
        minX = Math.min(minX, cx - r);
        minY = Math.min(minY, cy - r);
        maxX = Math.max(maxX, cx + r);
        maxY = Math.max(maxY, cy + r);
    });
    
    // Add padding
    const padding = 50;
    minX -= padding;
    minY -= padding;
    maxX += padding;
    maxY += padding;
    
    // Calculate width and height of the content
    const width = maxX - minX;
    const height = maxY - minY;
    
    // Calculate scale to fit the content
    const scale = Math.min(
        config.width / width,
        config.height / height,
        1.5 // Limit max zoom out for better visibility
    );
    
    // Calculate translate to center the content
    const centerX = (minX + maxX) / 2;
    const centerY = (minY + maxY) / 2;
    
    const translateX = config.width / 2 - scale * centerX;
    const translateY = config.height / 2 - scale * centerY;
    
    // Apply the transform
    const transform = d3.zoomIdentity
        .translate(translateX, translateY)
        .scale(scale);
    
    svg.transition()
      .duration(750)
      .call(zoom.transform, transform);
}

// Update graph parameters
function updateGraphParameters() {
    if (!state.simulation) return;
    
    // Update charge
    config.charge = parseInt(chargeSlider.value);
    chargeValue.textContent = config.charge;
    state.simulation.force('charge').strength(config.charge);
    
    // Update link distance
    config.linkDistance = parseInt(linkDistanceSlider.value);
    linkDistanceValue.textContent = config.linkDistance;
    state.simulation.force('link').distance(config.linkDistance);
    
    // Restart simulation
    state.simulation.alpha(0.3).restart();
}

// Apply edge filter and rebuild the graph
function updateEdgeFilter() {
    // Update edge filter percentage
    config.edgeFilterPercentage = parseInt(edgeFilterSlider.value);
    edgeFilterValue.textContent = config.edgeFilterPercentage + '%';
    
    // Apply filtering
    filterLinks();
    
    // Rebuild the graph with filtered links
    initForceGraph();
}

// Drag functions
function dragstarted(event, d) {
    if (!event.active) state.simulation.alphaTarget(0.3).restart();
    d.fx = d.x;
    d.fy = d.y;
}

function dragged(event, d) {
    d.fx = event.x;
    d.fy = event.y;
}

function dragended(event, d) {
    if (!event.active) state.simulation.alphaTarget(0);
    d.fx = null;
    d.fy = null;
}

// Handle window resize
function handleResize() {
    initGraphDimensions();
    if (state.simulation) {
        state.simulation.force('center', d3.forceCenter(config.width / 2, config.height / 2));
        state.simulation.alpha(0.3).restart();
    }
}

// Event listeners
edgeSelect.addEventListener('change', async (e) => {
    state.edgeFile = e.target.value;
    await loadLinks(state.edgeFile);
    initForceGraph();
});

chargeSlider.addEventListener('input', () => {
    chargeValue.textContent = chargeSlider.value;
});

chargeSlider.addEventListener('change', updateGraphParameters);

linkDistanceSlider.addEventListener('input', () => {
    linkDistanceValue.textContent = linkDistanceSlider.value;
});

linkDistanceSlider.addEventListener('change', updateGraphParameters);

edgeFilterSlider.addEventListener('input', () => {
    edgeFilterValue.textContent = edgeFilterSlider.value + '%';
});

edgeFilterSlider.addEventListener('change', updateEdgeFilter);

window.addEventListener('resize', handleResize);

// Initialize the graph
function init() {
    initGraphDimensions();
    loadData();
}

// Start app
init();

// Fetch available edge files and populate dropdown
async function loadEdgeFileOptions() {
    try {
        // Fetch the edge_files.json which contains a list of available edge files
        const response = await fetch('force_graph/edge_files.json');
        if (!response.ok) {
            throw new Error('Could not fetch edge files list');
        }
        
        const edgeFiles = await response.json();
        
        // Clear existing options
        edgeSelect.innerHTML = '';
        
        // Add options to the dropdown
        if (edgeFiles.length > 0) {
            edgeFiles.forEach(file => {
                const option = document.createElement('option');
                option.value = file;
                // Create a more readable name by removing .json and edges_ prefix and replacing underscores with spaces
                option.textContent = file.replace('.json', '').replace('edges_', '').replace(/_/g, ' ');
                edgeSelect.appendChild(option);
            });
            
            // Set the initial edge file
            state.edgeFile = edgeFiles[0];
        } else {
            // Fallback to default if no files found
            const option = document.createElement('option');
            option.value = 'edges_nQi_typing_speed.json';
            option.textContent = 'nQi typing speed';
            edgeSelect.appendChild(option);
            
            state.edgeFile = 'edges_nQi_typing_speed.json';
        }
    } catch (error) {
        console.error('Error loading edge file options:', error);
        
        // Fallback to default
        edgeSelect.innerHTML = '';
        const option = document.createElement('option');
        option.value = 'edges_nQi_typing_speed.json';
        option.textContent = 'nQi typing speed';
        edgeSelect.appendChild(option);
        
        state.edgeFile = 'edges_nQi_typing_speed.json';
    }
} 