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

    const { dims, tiles, limits } = VI_DATA;
    // dims: { M, K, N }
    // A: M x K
    // B: K x N
    // C: M x N
    
    // Canvas dimensions (Rows x Cols)
    // Canvas A: M rows x K cols
    // Canvas B: K rows x N cols
    // Canvas C: M rows x N cols
    
    const canvasA = document.getElementById('canvasA');
    const canvasB = document.getElementById('canvasB');
    const canvasC = document.getElementById('canvasC');
    
    const ctxA = canvasA.getContext('2d', { alpha: false });
    const ctxB = canvasB.getContext('2d', { alpha: false });
    const ctxC = canvasC.getContext('2d', { alpha: false });

    const sliderBlockCol = document.getElementById('blockCol'); // C col index (block_id.x)
    const sliderBlockRow = document.getElementById('blockRow'); // C row index (block_id.y)
    const sliderBlockK = document.getElementById('blockK');     // K index
    const sliderThreadID = document.getElementById('threadID');
    const sliderScale = document.getElementById('scale');
    
    const valBlockCol = document.getElementById('valBlockCol');
    const valBlockRow = document.getElementById('valBlockRow');
    const valBlockK = document.getElementById('valBlockK');
    const valThreadID = document.getElementById('valThreadID');
    const valScale = document.getElementById('valScale');

    // Init sliders
    sliderBlockCol.max = limits.max_block_col;
    sliderBlockRow.max = limits.max_block_row;
    sliderBlockK.max = limits.max_block_k;
    sliderThreadID.max = limits.max_thread_id;
    
    let currentScale = parseInt(sliderScale.value);

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

    function highlightTile(ctx, tile, color) {
        if (!tile) return;
        const px = tile.x * currentScale;
        const py = tile.y * currentScale;
        
        // Clamp dimensions to canvas bounds
        // The tile w/h are in logical units, scale them.
        let pw = tile.w * currentScale;
        let ph = tile.h * currentScale;
        
        // Canvas dims are also scaled
        const maxW = ctx.canvas.width - px;
        const maxH = ctx.canvas.height - py;
        
        if (pw > maxW) pw = maxW;
        if (ph > maxH) ph = maxH;
        
        if (pw <= 0 || ph <= 0) return;

        ctx.fillStyle = color;
        ctx.fillRect(px, py, pw, ph);

        ctx.strokeStyle = color.replace('0.5', '1.0').replace('0.7', '1.0');
        ctx.lineWidth = 2;
        ctx.strokeRect(px, py, pw, ph);
    }

    function update() {
        const bx = parseInt(sliderBlockCol.value); // C Col
        const by = parseInt(sliderBlockRow.value); // C Row
        const bk = parseInt(sliderBlockK.value);   // K
        const tx = parseInt(sliderThreadID.value);
        
        valBlockCol.textContent = bx;
        valBlockRow.textContent = by;
        valBlockK.textContent = bk;
        valThreadID.textContent = tx;
        
        // Draw Bases
        if (cacheA) ctxA.drawImage(cacheA, 0, 0);
        if (cacheB) ctxB.drawImage(cacheB, 0, 0);
        if (cacheC) ctxC.drawImage(cacheC, 0, 0);

        // Highlights
        
        // A: Key "row_k" -> "by_bk"
        // A's block is determined by C-row (by) and K-loop (bk)
        const aKey = `${by}_${bk}`;
        const aTile = tiles.A[aKey];
        if (aTile) highlightTile(ctxA, aTile, 'rgba(40, 167, 69, 0.5)'); // Green

        // B: Key "k_col" -> "bk_bx"
        // B's block is determined by K-loop (bk) and C-col (bx)
        const bKey = `${bk}_${bx}`;
        const bTile = tiles.B[bKey];
        if (bTile) highlightTile(ctxB, bTile, 'rgba(40, 167, 69, 0.5)'); // Green

        // C Block: "bx_by"
        const cBlockKey = `${bx}_${by}`;
        const cBlockTile = tiles.block[cBlockKey];
        if (cBlockTile) highlightTile(ctxC, cBlockTile, 'rgba(0, 123, 255, 0.5)'); // Blue

        // C Thread: "bx_by_tx"
        const cThreadKey = `${bx}_${by}_${tx}`;
        const cThreadTile = tiles.thread[cThreadKey];
        if (cThreadTile) highlightTile(ctxC, cThreadTile, 'rgba(255, 193, 7, 0.7)'); // Yellow
    }

    sliderBlockCol.addEventListener('input', update);
    sliderBlockRow.addEventListener('input', update);
    sliderBlockK.addEventListener('input', update);
    sliderThreadID.addEventListener('input', update);
    
    sliderScale.addEventListener('input', (e) => {
        currentScale = parseInt(e.target.value);
        valScale.textContent = currentScale;
        resizeCanvases();
        update();
    });

    resizeCanvases();
    update();
});