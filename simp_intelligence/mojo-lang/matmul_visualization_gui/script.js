document.addEventListener('DOMContentLoaded', async () => {
    const loadingEl = document.getElementById('loading');
    const mainContent = document.getElementById('mainContent');
    
    let VI_DATA = null;
    try {
        const response = await fetch('/data');
        if (!response.ok) throw new Error('Network response was not ok');
        VI_DATA = await response.json();
        
        loadingEl.style.display = 'none';
        mainContent.style.display = 'flex';
        
    } catch (error) {
        loadingEl.textContent = "Error loading data. Is the server running?";
        console.error(error);
        return;
    }

    const { dims, tiles, limits, steps } = VI_DATA;
    
    const canvasA = document.getElementById('canvasA');
    const canvasB = document.getElementById('canvasB');
    const canvasC = document.getElementById('canvasC');
    
    const ctxA = canvasA.getContext('2d', { alpha: false });
    const ctxB = canvasB.getContext('2d', { alpha: false });
    const ctxC = canvasC.getContext('2d', { alpha: false });

    const sliderBlockCol = document.getElementById('blockCol'); 
    const sliderBlockRow = document.getElementById('blockRow'); 
    const sliderBlockK = document.getElementById('blockK');     
    const sliderMmaK = document.getElementById('mmaK');
    const sliderMmaM = document.getElementById('mmaM');
    const sliderMmaN = document.getElementById('mmaN');
    const sliderThreadID = document.getElementById('threadID');
    const sliderScale = document.getElementById('scale');
    
    const valBlockCol = document.getElementById('valBlockCol');
    const valBlockRow = document.getElementById('valBlockRow');
    const valBlockK = document.getElementById('valBlockK');
    const valMmaK = document.getElementById('valMmaK');
    const valMmaM = document.getElementById('valMmaM');
    const valMmaN = document.getElementById('valMmaN');
    const valThreadID = document.getElementById('valThreadID');
    const valWarpID = document.getElementById('valWarpID');
    const valScale = document.getElementById('valScale');

    // Init sliders
    sliderBlockCol.max = limits.max_block_col;
    sliderBlockRow.max = limits.max_block_row;
    sliderBlockK.max = limits.max_block_k;
    
    sliderMmaK.max = steps.k_steps - 1;
    sliderMmaM.max = steps.m_steps - 1;
    sliderMmaN.max = steps.n_steps - 1;

    sliderThreadID.max = limits.max_thread_id;
    
    let currentScale = parseFloat(sliderScale.value);
    valScale.textContent = currentScale;

    let cacheA = null;
    let cacheB = null;
    let cacheC = null;

    function createGrid(rows, cols) {
        const c = document.createElement('canvas');
        c.width = cols * currentScale;
        c.height = rows * currentScale;
        const ctx = c.getContext('2d', { alpha: false });

        ctx.fillStyle = '#ffffff';
        ctx.fillRect(0, 0, c.width, c.height);

        ctx.beginPath();
        ctx.strokeStyle = '#e0e0e0';
        ctx.lineWidth = 1;

        // Verticals
        for (let x = 0; x <= cols; x++) {
            const px = Math.floor(x * currentScale) + 0.5;
            ctx.moveTo(px, 0);
            ctx.lineTo(px, c.height);
        }
        // Horizontals
        for (let y = 0; y <= rows; y++) {
            const py = Math.floor(y * currentScale) + 0.5;
            ctx.moveTo(0, py);
            ctx.lineTo(c.width, py);
        }
        ctx.stroke();
        return c;
    }

    function resizeCanvases() {
        // A: M x K
        canvasA.width = dims.K * currentScale;
        canvasA.height = dims.M * currentScale;
        cacheA = createGrid(dims.M, dims.K);

        // B: K x N
        canvasB.width = dims.N * currentScale;
        canvasB.height = dims.K * currentScale;
        cacheB = createGrid(dims.K, dims.N);

        // C: M x N
        canvasC.width = dims.N * currentScale;
        canvasC.height = dims.M * currentScale;
        cacheC = createGrid(dims.M, dims.N);
    }

    function highlightTile(ctx, tile, color, lineWidth = 2) {
        if (!tile) return;
        
        // Server: tile.x = Row (Canvas Y), tile.y = Col (Canvas X)
        // tile.w = Cols, tile.h = Rows
        
        const px = tile.y * currentScale; // Col -> X
        const py = tile.x * currentScale; // Row -> Y
        
        let pw = tile.w * currentScale;
        let ph = tile.h * currentScale;
        
        const maxW = ctx.canvas.width - px;
        const maxH = ctx.canvas.height - py;
        
        // Clip visual overflow
        if (pw > maxW) pw = maxW;
        if (ph > maxH) ph = maxH;
        
        if (pw <= 0 || ph <= 0) return;

        ctx.fillStyle = color;
        ctx.fillRect(px, py, pw, ph);

        // Darker border
        ctx.strokeStyle = color.replace(/[\d.]+\)$/, '1.0)'); 
        ctx.lineWidth = lineWidth;
        ctx.strokeRect(px, py, pw, ph);
    }

    function update() {
        const bx = parseInt(sliderBlockCol.value); 
        const by = parseInt(sliderBlockRow.value); 
        const bk = parseInt(sliderBlockK.value);
        
        const mmaK = parseInt(sliderMmaK.value);
        const mmaM = parseInt(sliderMmaM.value);
        const mmaN = parseInt(sliderMmaN.value);
        
        // step = mma_k * (m_steps * n_steps) + mma_m * (n_steps) + mma_n
        const step = mmaK * (steps.m_steps * steps.n_steps) + mmaM * steps.n_steps + mmaN;
        
        const tx = parseInt(sliderThreadID.value);
        
        const warpId = Math.floor(tx / dims.WARP_SIZE);
        
        valBlockCol.textContent = bx;
        valBlockRow.textContent = by;
        valBlockK.textContent = bk;
        
        valMmaK.textContent = mmaK;
        valMmaM.textContent = mmaM;
        valMmaN.textContent = mmaN;
        
        valThreadID.textContent = tx;
        valWarpID.textContent = warpId;
        
        // Draw Bases
        if (cacheA) ctxA.drawImage(cacheA, 0, 0);
        if (cacheB) ctxB.drawImage(cacheB, 0, 0);
        if (cacheC) ctxC.drawImage(cacheC, 0, 0);

        // Highlights
        
        // 1. Block Tiles (Shared Mem load)
        // A: Key "by_bk"
        const aKey = `${by}_${bk}`;
        const aTile = tiles.A_tile[aKey];
        if (aTile) highlightTile(ctxA, aTile, 'rgba(40, 167, 69, 0.3)'); // Green

        // B: Key "bk_bx"
        const bKey = `${bk}_${bx}`;
        const bTile = tiles.B_tile[bKey];
        if (bTile) highlightTile(ctxB, bTile, 'rgba(40, 167, 69, 0.3)'); // Green
        
        // 2. Warp Tiles (Shared -> Reg/Warp)
        // Key: bx_by_tx_bk
        // Note: multiple threads in same warp map to same warp tile.
        // We use tx directly as key because log has thread_id.
        const warpKey = `${bx}_${by}_${tx}_${bk}`;
        const aWarp = tiles.A_warp[warpKey];
        const bWarp = tiles.B_warp[warpKey];
        
        if (aWarp) highlightTile(ctxA, aWarp, 'rgba(0, 123, 255, 0.4)'); // Blue
        if (bWarp) highlightTile(ctxB, bWarp, 'rgba(0, 123, 255, 0.4)'); // Blue
        
        // 3. MMA Tiles (Reg -> Compute)
        // Key: bx_by_tx
        // Value: Array of tiles. Index = bk * steps_per_k + step
        const mmaKey = `${bx}_${by}_${tx}`;
        const globalStep = bk * steps.total_per_k + step;
        
        const getMmaTile = (collection) => {
            const list = collection[mmaKey];
            if (list && list[globalStep]) return list[globalStep];
            return null;
        };

        const aMma = getMmaTile(tiles.A_mma);
        const bMma = getMmaTile(tiles.B_mma);
        const cMma = getMmaTile(tiles.C_mma);
        
        if (aMma) highlightTile(ctxA, aMma, 'rgba(220, 53, 69, 0.7)'); // Red
        if (bMma) highlightTile(ctxB, bMma, 'rgba(220, 53, 69, 0.7)'); // Red
        if (cMma) highlightTile(ctxC, cMma, 'rgba(255, 193, 7, 0.7)'); // Yellow
        
        // Also highlight C Block (Result) - Not logged explicitly?
        // Usually C block is M x N block corresponding to bx, by.
        // dims.BM x dims.BN.
        // x (col) = bx * BN. y (row) = by * BM.
        const cBlockTile = { 
            x: by * dims.BM, // Row
            y: bx * dims.BN, // Col
            w: dims.BN, 
            h: dims.BM 
        };
        // Draw outline for C block
        // highlightTile(ctxC, cBlockTile, 'rgba(0,0,0,0.1)', 1);
        // Maybe just a subtle border
    }

    sliderBlockCol.addEventListener('input', update);
    sliderBlockRow.addEventListener('input', update);
    sliderBlockK.addEventListener('input', update);
    sliderMmaK.addEventListener('input', update);
    sliderMmaM.addEventListener('input', update);
    sliderMmaN.addEventListener('input', update);
    sliderThreadID.addEventListener('input', update);
    
    sliderScale.addEventListener('input', (e) => {
        currentScale = parseFloat(e.target.value);
        valScale.textContent = currentScale;
        resizeCanvases();
        update();
    });

    resizeCanvases();
    update();
});
