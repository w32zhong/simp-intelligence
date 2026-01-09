const http = require('http');
const fs = require('fs');
const path = require('path');
const readline = require('readline');

const PORT = 3000;
const LOG_FILES = {
    block: path.join(__dirname, '../block_tile.log'),
    thread: path.join(__dirname, '../thread_tile.log'),
    A: path.join(__dirname, '../A_tile.log'),
    B: path.join(__dirname, '../B_tile.log'),
    A_sub: path.join(__dirname, '../A_subtile.log'),
    B_sub: path.join(__dirname, '../B_element.log')
};
const MOJO_FILE = path.join(__dirname, '../matmul_visualization.mojo');

let cachedData = null;

async function getMatrixDimensions() {
    let dims = { M: 0, K: 0, N: 0, BM: 4, BN: 4, BK: 4 };

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
    const aSubTileMap = {};
    const bSubTileMap = {};
    
    let maxBlockCol = 0; 
    let maxBlockRow = 0; 
    let maxThreadID = 0;
    let maxBlockK = 0;
    let maxInnerK = 0;

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

                // Default tile (block/thread/A/B)
                let tile = {
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
                } else if (type === 'A_sub') {
                    const k = entry['block'];
                    const innerK = entry['k'];
                    const tx = entry['thread_id.x'];
                    maxInnerK = Math.max(maxInnerK, innerK);
                    // Key: Row(by) _ BlockK(k) _ InnerK(innerK) _ Thread(tx)
                    aSubTileMap[`${by}_${k}_${innerK}_${tx}`] = tile;
                } else if (type === 'B_sub') {
                    const k = entry['block'];
                    const innerK = entry['k'];
                    const col = entry['col'];
                    const tx = entry['thread_id.x'];
                    maxInnerK = Math.max(maxInnerK, innerK);
                    
                    // Override tile for single element
                    tile = {
                        x: entry['y'] + col,
                        y: entry['x'] + innerK,
                        w: 1,
                        h: 1
                    };
                    
                    // Key: BlockK(k) _ Col(bx) _ InnerK(innerK) _ Thread(tx)
                    bSubTileMap[`${k}_${bx}_${innerK}_${tx}`] = tile;
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
    await processFile(LOG_FILES.A_sub, 'A_sub');
    await processFile(LOG_FILES.B_sub, 'B_sub');

    const dims = await getMatrixDimensions();
    
    if (dims.M === 0) dims.M = 480;
    if (dims.K === 0) dims.K = 560;
    if (dims.N === 0) dims.N = 240;

    const theoreticalMaxCol = Math.ceil(dims.N / dims.BN) - 1;
    const theoreticalMaxRow = Math.ceil(dims.M / dims.BM) - 1;
    const theoreticalMaxK = Math.ceil(dims.K / dims.BK) - 1;
    const theoreticalMaxInnerK = dims.BK - 1;

    const finalMaxCol = theoreticalMaxCol; 
    const finalMaxRow = theoreticalMaxRow;
    const finalMaxK = theoreticalMaxK;
    const finalMaxInnerK = theoreticalMaxInnerK;

    console.log(`Limits Enforced: Col:${finalMaxCol}, Row:${finalMaxRow}, K:${finalMaxK}, InnerK:${finalMaxInnerK}`);

    return {
        dims: dims, 
        tiles: {
            block: blockTileMap,
            thread: threadTileMap,
            A: aTileMap,
            B: bTileMap,
            A_sub: aSubTileMap,
            B_sub: bSubTileMap
        },
        limits: {
            max_block_col: finalMaxCol,
            max_block_row: finalMaxRow,
            max_thread_id: maxThreadID,
            max_block_k: finalMaxK,
            max_inner_k: finalMaxInnerK
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
