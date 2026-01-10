const http = require('http');
const fs = require('fs');
const path = require('path');
const readline = require('readline');

const PORT = 3000;
const LOG_FILES = {
    A_tile: path.join(__dirname, '../A_tile.log'),
    B_tile: path.join(__dirname, '../B_tile.log'),
    A_warp: path.join(__dirname, '../A_warp_tile.log'),
    B_warp: path.join(__dirname, '../B_warp_tile.log'),
    A_mma: path.join(__dirname, '../A_mma_tile.log'),
    B_mma: path.join(__dirname, '../B_mma_tile.log'),
    C_mma: path.join(__dirname, '../C_mma_tile.log')
};
const MOJO_FILE = path.join(__dirname, '../matmul_visualization.mojo');

let cachedData = null;

async function getMatrixDimensions() {
    let dims = { 
        M: 0, K: 0, N: 0, 
        BM: 16, BN: 16, BK: 16, 
        WM: 8, WN: 8, 
        MMA_M: 4, MMA_N: 4, MMA_K: 2,
        NUM_WARPS: 4, WARP_SIZE: 8
    };

    if (fs.existsSync(MOJO_FILE)) {
        const content = fs.readFileSync(MOJO_FILE, 'utf-8');
        
        const parse = (name, def) => {
            const re = new RegExp(`(alias|comptime)\\s+${name}\\s*=\s*(\\d+|OPTIMIZED_BLOCK_SIZE)`);
            const match = content.match(re);
            if (!match) return def;
            if (match[2] === 'OPTIMIZED_BLOCK_SIZE') {
                const optMatch = content.match(/comptime\s+OPTIMIZED_BLOCK_SIZE\s*=\s*(\d+)/);
                return optMatch ? parseInt(optMatch[1], 10) : 16;
            }
            return parseInt(match[2], 10);
        };
        
        // Complex parser for math expressions like (BM // WM) * ... is hard with regex. 
        // We will assume simple integers or variable references for now, or default values.
        
        dims.M = parse('M', 64);
        dims.K = parse('K', 48);
        dims.N = parse('N', 64);
        
        dims.BM = parse('BM', 16);
        dims.BN = parse('BN', 16);
        dims.BK = parse('BK', 16);
        
        dims.WM = parse('WM', 8);
        dims.WN = parse('WN', 8);
        
        dims.MMA_M = parse('MMA_M', 4);
        dims.MMA_N = parse('MMA_N', 4);
        dims.MMA_K = parse('MMA_K', 2);
        
        // Calculate NUM_WARPS if not explicitly parsed (simple heuristic)
        // comptime NUM_WARPS = (BM // WM) * (BN // WN)
        dims.NUM_WARPS = Math.floor(dims.BM / dims.WM) * Math.floor(dims.BN / dims.WN);

        console.log(`Parsed dimensions:`, dims);
    } else {
        console.warn("Mojo file not found.");
    }
    return dims;
}

async function parseLogs() {
    console.log("Parsing logs...");
    const tiles = {
        A_tile: {},
        B_tile: {},
        A_warp: {},
        B_warp: {},
        A_mma: {},
        B_mma: {},
        C_mma: {}
    };
    
    let limits = {
        max_block_col: 0,
        max_block_row: 0,
        max_block_k: 0,
        max_thread_id: 0
    };

    const processFile = async (filePath, type, storage) => {
        if (!fs.existsSync(filePath)) {
            console.warn(`File not found: ${filePath}`);
            return;
        }

        const fileStream = fs.createReadStream(filePath);
        const rl = readline.createInterface({
            input: fileStream,
            crlfDelay: Infinity
        });

        for await (const line of rl) {
            try {
                if (!line.trim()) continue;
                const entry = JSON.parse(line);
                
                const bx = entry['block_id.x']; 
                const by = entry['block_id.y'];
                const tx = entry['thread_id.x']; // Assuming 1D thread block for visualization
                
                limits.max_block_col = Math.max(limits.max_block_col, bx);
                limits.max_block_row = Math.max(limits.max_block_row, by);
                limits.max_thread_id = Math.max(limits.max_thread_id, tx);

                let tile = {
                    x: entry['x'], 
                    y: entry['y'], 
                    w: entry['n_cols'], 
                    h: entry['n_rows']
                };

                if (type === 'A_tile' || type === 'B_tile') {
                    // Key: by_k for A, k_bx for B
                    const k = entry['block'];
                    limits.max_block_k = Math.max(limits.max_block_k, k);
                    
                    const key = type === 'A_tile' ? `${by}_${k}` : `${k}_${bx}`;
                    storage[key] = tile;
                } 
                else if (type === 'A_warp' || type === 'B_warp') {
                    // Key: bx_by_tx_k (Need k because it changes per block loop)
                    const k = entry['block'];
                    const key = `${bx}_${by}_${tx}_${k}`;
                    storage[key] = tile;
                }
                else if (type.includes('mma')) {
                    // List of tiles for a given thread/block
                    const key = `${bx}_${by}_${tx}`;
                    if (!storage[key]) storage[key] = [];
                    storage[key].push(tile);
                }

            } catch (e) {
                // Ignore malformed lines
            }
        }
    };

    await processFile(LOG_FILES.A_tile, 'A_tile', tiles.A_tile);
    await processFile(LOG_FILES.B_tile, 'B_tile', tiles.B_tile);
    await processFile(LOG_FILES.A_warp, 'A_warp', tiles.A_warp);
    await processFile(LOG_FILES.B_warp, 'B_warp', tiles.B_warp);
    await processFile(LOG_FILES.A_mma, 'A_mma', tiles.A_mma);
    await processFile(LOG_FILES.B_mma, 'B_mma', tiles.B_mma);
    await processFile(LOG_FILES.C_mma, 'C_mma', tiles.C_mma);

    const dims = await getMatrixDimensions();
    
    // Fallback/Validation
    if (dims.M === 0) dims.M = 64;
    
    // Calculate max steps for MMA
    // loops: k (BK/MMA_K) -> m (WM/MMA_M) -> n (WN/MMA_N)
    const k_steps = Math.floor(dims.BK / dims.MMA_K);
    const m_steps = Math.floor(dims.WM / dims.MMA_M);
    const n_steps = Math.floor(dims.WN / dims.MMA_N);
    const mma_steps_per_k = m_steps * n_steps; 
    // Note: The loop order in mojo is k, m, n. 
    // Total steps recorded per K-block = mma_steps_per_k * k_steps ?? 
    // Wait, mojo loop:
    // for mma_k...
    //   for mma_m...
    //     for mma_n...
    //       log...
    // So for one K-block (one 'block' iteration), we have k_steps * m_steps * n_steps entries.
    
    return {
        dims: dims, 
        tiles: tiles,
        limits: limits,
        steps: {
            k_steps, m_steps, n_steps, total_per_k: k_steps * m_steps * n_steps
        }
    };
}

parseLogs().then(data => {
    cachedData = data;
});

const server = http.createServer((req, res) => {
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET');

    if (req.url === '/data') {
        if (!cachedData) {
            res.writeHead(503, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ error: 'Data loading...' }));
        } else {
            res.writeHead(200, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify(cachedData));
        }
        return;
    }

    let filePath = path.join(__dirname, req.url === '/' ? 'index.html' : req.url);
    const extname = path.extname(filePath);
    let contentType = 'text/html';

    switch (extname) {
        case '.js': contentType = 'text/javascript'; break;
        case '.css': contentType = 'text/css'; break;
        case '.json': contentType = 'application/json'; break;
    }

    fs.readFile(filePath, (err, content) => {
        if (err) {
            if(err.code == 'ENOENT') {
                res.writeHead(404);
                res.end(`File not found: ${req.url}`);
            } else {
                res.writeHead(500);
                res.end(`Server Error: ${err.code}`);
            }
        } else {
            res.writeHead(200, { 'Content-Type': contentType });
            res.end(content, 'utf-8');
        }
    });
});

server.listen(PORT, () => {
    console.log(`Server running at http://localhost:${PORT}/`);
});
