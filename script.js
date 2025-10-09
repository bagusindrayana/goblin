const MODEL_URL = '/models';
const TARGET_IMAGE_URL = '/images/bahlil.jpg';

let faceMatcher = null;
let uploadedFiles = [];
let censorOption = 'pixelated';

// --- Elemen DOM ---
const inputImageUpload = document.getElementById('inputImageUpload');
const scanButton = document.getElementById('scanButton');
const statusMessage = document.getElementById('statusMessage');
const imageGridContainer = document.getElementById('imageGridContainer');
const overallMatchStatus = document.getElementById('overallMatchStatus');


function updateStatus(message, colorClass = 'text-green-400 animate-pulse') {
    statusMessage.textContent = `// ${message} //`;
    statusMessage.className = `mt-4 text-center text-sm ${colorClass}`;
}

async function initializeSystem() {
    updateStatus("Loading AI protocols...");
    try {
        await Promise.all([
            faceapi.nets.ssdMobilenetv1.loadFromUri(MODEL_URL),
            faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL),
            faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL)
        ]);
        await loadTargetFace();
    } catch (error) {
        updateStatus(`ERROR: System initialization failed. Check console.`, 'text-red-500');
        console.error("Initialization Error:", error);
    }
}

async function loadTargetFace() {
    updateStatus("Acquiring target subject data...");
    try {
        const targetImg = await faceapi.fetchImage(TARGET_IMAGE_URL);
        const detections = await faceapi.detectSingleFace(targetImg).withFaceLandmarks().withFaceDescriptor();
        if (detections) {
            const labeledFaceDescriptors = [new faceapi.LabeledFaceDescriptors('TARGET', [detections.descriptor])];
            faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors, 0.6);
            updateStatus("System ready. Upload images for scan.");
            inputImageUpload.disabled = false;
            scanButton.textContent = "INITIATE SCAN";
            scanButton.classList.add('animate-pulse-once');
        } else {
            throw new Error('No face detected in predefined target image.');
        }
    } catch (error) {
        updateStatus(`ERROR: Could not acquire target data.`, 'text-red-500');
        console.error("Error loading target face:", error);
    }
}


async function processImage(file) {
    updateStatus(`Analyzing: ${file.name}...`);
    
    const imageElement = await faceapi.bufferToImage(file);
    const canvas = faceapi.createCanvasFromMedia(imageElement);
    const displaySize = { width: imageElement.width, height: imageElement.height };
    faceapi.matchDimensions(canvas, displaySize);
    const context = canvas.getContext('2d');
    context.drawImage(imageElement, 0, 0, displaySize.width, displaySize.height);

    try {
        const detections = await faceapi.detectAllFaces(imageElement).withFaceLandmarks().withFaceDescriptors();
        const resizedDetections = faceapi.resizeResults(detections, displaySize);
        let foundTargetInImage = false;

        resizedDetections.forEach(d => {
            const bestMatch = faceMatcher.findBestMatch(d.descriptor);
            const box = d.detection.box;

            if (bestMatch.label === 'TARGET') {
                foundTargetInImage = true;

                if (censorOption === 'black') {
                    applyBlackBoxCensor(context, box);
                } else {
                    applyPixelatedCensor(context, imageElement, box);
                }
            }
    });
    return { canvas, foundTarget: foundTargetInImage, fileName: file.name };
    } catch (error) {
        console.error(`Error processing image ${file.name}:`, error);
        return { canvas: null, foundTarget: false, fileName: file.name, error: true };
    }
}

// black box censor
function applyBlackBoxCensor(context, box) {
    context.save();
    context.fillStyle = 'black';
    context.fillRect(box.x, box.y, box.width, box.height);
    context.restore();
}
// Pixelate censor
function applyPixelatedCensor(context, imageElement, box) {
    // Adaptive pixelation
    const x = Math.max(0, Math.floor(box.x));
    const y = Math.max(0, Math.floor(box.y));
    const width = Math.max(1, Math.floor(box.width));
    const height = Math.max(1, Math.floor(box.height));

    const faceSize = Math.max(width, height);
    const BASE_BLOCK_RATIO = 0.12;
    let pixelBlock = Math.max(2, Math.round(faceSize * BASE_BLOCK_RATIO));

    const imageMaxDim = Math.max(imageElement.width || width, imageElement.height || height);
    const maxBlock = Math.max(2, Math.round(imageMaxDim * 0.08));
    pixelBlock = Math.min(pixelBlock, maxBlock);

    const scaledW = Math.max(1, Math.ceil(width / pixelBlock));
    const scaledH = Math.max(1, Math.ceil(height / pixelBlock));

    const off = document.createElement('canvas');
    off.width = width;
    off.height = height;
    const offCtx = off.getContext('2d');
    offCtx.drawImage(imageElement, x, y, width, height, 0, 0, width, height);

    // tiny canvas for pixelation
    const tiny = document.createElement('canvas');
    tiny.width = scaledW;
    tiny.height = scaledH;
    const tinyCtx = tiny.getContext('2d');
    tinyCtx.imageSmoothingEnabled = false;
    tinyCtx.clearRect(0, 0, scaledW, scaledH);
    tinyCtx.drawImage(off, 0, 0, width, height, 0, 0, scaledW, scaledH);

    context.save();
    context.imageSmoothingEnabled = false;
    context.drawImage(tiny, 0, 0, scaledW, scaledH, x, y, width, height);
    context.restore();
}

const _censorRadios = document.querySelectorAll('input[name="censorOption"]');
_censorRadios.forEach(r => r.addEventListener('change', (e) => {
    censorOption = e.target.value;
}));

inputImageUpload.addEventListener('change', (event) => {
    uploadedFiles = Array.from(event.target.files);
    if (uploadedFiles.length > 0) {
        scanButton.disabled = false;
        scanButton.textContent = `SCAN ${uploadedFiles.length} IMAGES`;
        overallMatchStatus.textContent = '';
        imageGridContainer.innerHTML = '';
        updateStatus(`// ${uploadedFiles.length} images loaded. Ready to scan. //`);
    } else {
        scanButton.disabled = true;
        scanButton.textContent = "INITIATE SCAN";
    }
});

scanButton.addEventListener('click', async () => {
    if (!faceMatcher || uploadedFiles.length === 0) return;

    updateStatus("Initiating batch scan...");
    scanButton.disabled = true;
    imageGridContainer.innerHTML = '';
    overallMatchStatus.textContent = '';
    let totalTargetsFound = 0;

    for (const file of uploadedFiles) {
        const result = await processImage(file);
        

        const gridItem = document.createElement('div');
        gridItem.className = 'result-item relative bg-gray-700 border border-gray-600 rounded-md overflow-hidden p-2 flex flex-col items-center justify-center';

        if (result.canvas) {
            result.canvas.style.maxWidth = '100%';
            result.canvas.style.height = 'auto';
            gridItem.appendChild(result.canvas);


            const downloadBtn = document.createElement('a');
            // Konversi canvas ke data URL
            downloadBtn.href = result.canvas.toDataURL('image/png'); 
            downloadBtn.download = `censored_${result.fileName}`;
            downloadBtn.className = 'download-btn absolute top-2 right-2 bg-green-600 hover:bg-green-700 text-white p-2 rounded-full transition-all duration-300 shadow-lg';
            downloadBtn.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M3 17a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm3.293-7.707a1 1 0 011.414 0L9 10.586V3a1 1 0 112 0v7.586l1.293-1.293a1 1 0 111.414 1.414l-3 3a1 1 0 01-1.414 0l-3-3a1 1 0 010-1.414z" clip-rule="evenodd" /></svg>`;
            gridItem.appendChild(downloadBtn);
        }
        
        const fileNameLabel = document.createElement('p');
        fileNameLabel.className = 'text-xs text-gray-400 mt-2 truncate w-full px-2 text-center';
        fileNameLabel.textContent = result.fileName;
        gridItem.appendChild(fileNameLabel);

        const imageStatus = document.createElement('p');
        imageStatus.className = 'text-sm font-bold mt-1';
        if (result.foundTarget) {
            imageStatus.textContent = 'TARGET MATCHED';
            imageStatus.classList.add('text-green-400', 'drop-shadow-neon-green-sm');
            totalTargetsFound++;
        } else {
            imageStatus.textContent = 'NO TARGET';
            imageStatus.classList.add('text-yellow-400');
        }
        gridItem.appendChild(imageStatus);

        imageGridContainer.appendChild(gridItem);
    }

    // Perbarui status keseluruhan
    if (totalTargetsFound > 0) {
        updateStatus(`Scan complete. ${totalTargetsFound} target(s) identified.`, 'text-green-400');
        overallMatchStatus.textContent = `>> ${totalTargetsFound} TARGET(S) CONFIRMED <<`;
        overallMatchStatus.className = "mt-8 text-center text-xl font-bold text-green-400 drop-shadow-neon-green";
    } else {
        updateStatus("Scan complete. No targets detected.", 'text-yellow-400');
        overallMatchStatus.textContent = ">> NO TARGETS DETECTED <<";
        overallMatchStatus.className = "mt-8 text-center text-xl font-bold text-yellow-400";
    }

    scanButton.disabled = false;
    scanButton.textContent = "RESCAN";
});

// --- Inisialisasi ---
initializeSystem();