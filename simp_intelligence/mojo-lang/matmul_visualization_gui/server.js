const http = require('http');
const fs = require('fs');
const path = require('path');
const readline = require('readline');

const PORT = 3000;
const LOG_FILES = {
    block: path.join(__dirname, '../block_tile.log'),
    thread: path.join(__dirname, '../thread_tile.log'),
    A: path.join(__dirname, '../A_tile.log'),
    B: path.join(__dirname, '../B_tile.log')
};
const MOJO_FILE = path.join(__dirname, '../matmul_visualization.mojo');

let cachedData = null;

async function getMatrixDimensions() {
    let dims = { M: 0, K: 0, N: 0, BM: 4, BN: 4, BK: 4 }; // Defaults

    if (fs.existsSync(MOJO_FILE)) {
        const content = fs.readFileSync(MOJO_FILE, 'utf-8');
        const mMatch = content.match(/alias\s+M\s*=\s*(\d+)/);
        const kMatch = content.match(/alias\s+K\s*=\s*(\d+)/);
        const nMatch = content.match(/alias\s+N\s*=\s*(\d+)/);
        
        const bmMatch = content.match(/comptime\s+BM\s*=\s*(\d+|OPTIMIZED_BLOCK_SIZE)/);
        const bnMatch = content.match(/comptime\s+BN\s*=\s*(\d+|OPTIMIZED_BLOCK_SIZE)/);
        const bkMatch = content.match(/comptime\s+BK\s*=\s*(\d+|OPTIMIZED_BLOCK_SIZE)/);
        const optMatch = content.match(/comptime\s+OPTIMIZED_BLOCK_SIZE\s*=\s*(\d+)/);

        let optSize = 4;
        if (optMatch) optSize = parseInt(optMatch[1], 10);

        if (mMatch) dims.M = parseInt(mMatch[1], 10);
        if (kMatch) dims.K = parseInt(kMatch[1], 10);
        if (nMatch) dims.N = parseInt(nMatch[1], 10);

        // helper to resolve value
        const resolve = (match) => {
            if (!match) return 4;
            if (match[1] === 'OPTIMIZED_BLOCK_SIZE') return optSize;
            return parseInt(match[1], 10);
        };

        dims.BM = resolve(bmMatch);
        dims.BN = resolve(bnMatch);
        dims.BK = resolve(bkMatch);
        
        console.log(`Parsed dimensions: M=${dims.M}, K=${dims.K}, N=${dims.N}, BM=${dims.BM}, BK=${dims.BK}, BN=${dims.BN}`);
    } else {
        console.warn("Mojo file not found.");
    }
    return dims;
}

async function parseLogs() {
    console.log("Parsing logs...");
    const blockTileMap = {};
    const threadTileMap = {};
    const aTileMap = {};
    const bTileMap = {};
    
    let maxBlockCol = 0; 
    let maxBlockRow = 0; 
    let maxThreadID = 0;
    let maxBlockK = 0; 

    // Helper to process a file
    const processFile = async (filePath, type) => {
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
                
                maxBlockCol = Math.max(maxBlockCol, bx);
                maxBlockRow = Math.max(maxBlockRow, by);

                const tile = {
                    x: entry['y'], 
                    y: entry['x'], 
                    w: entry['n_cols'], 
                    h: entry['n_rows']
                };

                if (type === 'block') {
                    blockTileMap[`${bx}_${by}`] = tile;
                } else if (type === 'thread') {
                    const tx = entry['thread_id.x'];
                    maxThreadID = Math.max(maxThreadID, tx);
                    threadTileMap[`${bx}_${by}_${tx}`] = tile;
                } else if (type === 'A') {
                    const k = entry['block'];
                    maxBlockK = Math.max(maxBlockK, k);
                    aTileMap[`${by}_${k}`] = tile;
                } else if (type === 'B') {
                    const k = entry['block'];
                    maxBlockK = Math.max(maxBlockK, k);
                    bTileMap[`${k}_${bx}`] = tile;
                }

            } catch (e) {
                // Ignore malformed lines
            }
        }
    };

    await processFile(LOG_FILES.block, 'block');
    await processFile(LOG_FILES.thread, 'thread');
    await processFile(LOG_FILES.A, 'A');
    await processFile(LOG_FILES.B, 'B');

    const dims = await getMatrixDimensions();
    
    if (dims.M === 0) dims.M = 480;
    if (dims.K === 0) dims.K = 560;
    if (dims.N === 0) dims.N = 240;

    // Constrain limits based on parsed dimensions to handle stale logs
    // Grid dims are ceildiv(N, BN) and ceildiv(M, BM)
    const theoreticalMaxCol = Math.ceil(dims.N / dims.BN) - 1;
    const theoreticalMaxRow = Math.ceil(dims.M / dims.BM) - 1;
    const theoreticalMaxK = Math.ceil(dims.K / dims.BK) - 1;

    // Use minimum of log-observed and theoretical to be safe, 
    // OR just use theoretical to force consistency with current code version?
    // Using theoretical is safer for "input B goes off border" issue.
    
    const finalMaxCol = theoreticalMaxCol; 
    const finalMaxRow = theoreticalMaxRow;
    const finalMaxK = theoreticalMaxK;

    console.log(`Limits: Logs(Col:${maxBlockCol}, Row:${maxBlockRow}, K:${maxBlockK}) -> Enforced(Col:${finalMaxCol}, Row:${finalMaxRow}, K:${finalMaxK})`);

    return {
        dims: dims, 
        tiles: {
            block: blockTileMap,
            thread: threadTileMap,
            A: aTileMap,
            B: bTileMap
        },
        limits: {
            max_block_col: finalMaxCol,
            max_block_row: finalMaxRow,
            max_thread_id: maxThreadID, // Thread ID usually static
            max_block_k: finalMaxK
        }
    };
}

// Parse logs immediately on start
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