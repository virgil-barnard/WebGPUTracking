import { DeepSort } from './tracker.js';

const origWarn = console.warn;
console.warn = (...args) => {
  // filter out the one NMS warning
  if (
    typeof args[0] === 'string' &&
    args[0].includes('tf.nonMaxSuppression() in webgpu locks the UI thread')
  ) {
    return;
  }
  origWarn(...args);
};

const tf        = window.tf;
const cocoSsd   = window.cocoSsd;
const mobilenet = window.mobilenet;

const DETECT_EVERY = 3;
const SW = 320, SH = 240;
let frameCount = 0;
let lastTime = performance.now();

let video, canvas, ctx;
let smallCanvas, smallCtx;
let model, reidModel, tracker;

export async function startApp() {
  await tf.setBackend('webgpu');
  await tf.ready();

  model     = await cocoSsd.load({ base: 'lite_mobilenet_v2' });
  reidModel = await mobilenet.load({ version: 1, alpha: 0.25 });
  tracker   = new DeepSort(60, 0.45);

  video  = document.getElementById('video');
  canvas = document.getElementById('canvas');
  ctx    = canvas.getContext('2d');

  smallCanvas = document.createElement('canvas');
  smallCanvas.width = SW;
  smallCanvas.height = SH;
  smallCtx = smallCanvas.getContext('2d');

  const stream = await navigator.mediaDevices.getUserMedia({ video: true });
  video.srcObject = stream;
  await video.play();

  canvas.width  = video.videoWidth;
  canvas.height = video.videoHeight;

  requestAnimationFrame(detectLoop);
}

async function detectLoop() {
  frameCount = (frameCount + 1) % DETECT_EVERY;
  const now  = performance.now();
  const fps  = 1000 / (now - lastTime);
  lastTime   = now;

  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  let boxesArr = [], scoresArr = [], labelsArr = [], tracks;

  if (frameCount === 0) {
    smallCtx.drawImage(video, 0, 0, SW, SH);
    const predictions = await model.detect(smallCanvas);

    predictions.forEach(p => {
      const [x, y, w, h] = p.bbox;
      boxesArr.push([
        y * canvas.height / SH,
        x * canvas.width / SW,
        (y + h) * canvas.height / SH,
        (x + w) * canvas.width / SW
      ]);
      scoresArr.push(p.score);
      labelsArr.push(p.class);
    });

    const embeddings = [];
    for (let i = 0; i < boxesArr.length; i++) {
      const [y1, x1, y2, x2] = boxesArr[i];
      const y1n = y1 / canvas.height,
            x1n = x1 / canvas.width,
            y2n = y2 / canvas.height,
            x2n = x2 / canvas.width;

      const embTensor = tf.tidy(() => {
        const frame = tf.browser.fromPixels(video).expandDims(0);
        const crop = tf.image.cropAndResize(frame, [[ y1n, x1n, y2n, x2n ]], [0], [224, 224]);
        const emb = reidModel.infer(crop).squeeze();
        frame.dispose();
        crop.dispose();
        return emb;
      });

      const arr = await embTensor.array();
      embTensor.dispose();
      embeddings.push(arr);
    }

    tracks = await tracker.update(boxesArr, embeddings, labelsArr, scoresArr);
  } else {
    tracker.tracks.forEach(t => t.predict());
    tracks = tracker.tracks;
  }

  // Draw tracks
  ctx.strokeStyle = 'red';
  ctx.fillStyle = 'red';
  ctx.font = '14px monospace';
  ctx.lineWidth = 2;
  ctx.textBaseline = 'top';

  tracks.forEach(t => {
    const [y1, x1, y2, x2] = t.bbox;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    ctx.fillText(`ID:${t.id}`, x1 + 4, y1 + 4);
    if (t.label && typeof t.score === 'number') {
      ctx.fillText(`${t.label} ${(t.score * 100).toFixed(1)}%`, x1 + 4, y1 + 24);
    }
  });

  // Draw FPS + memory stats
  ctx.fillStyle = 'yellow';
  ctx.font      = '18px sans-serif';
  ctx.fillText(`FPS: ${fps.toFixed(1)}`, 10, 20);

  const mem   = tf.memory();
  const gpuMB = (mem.numBytes / 1024 / 1024).toFixed(2);
  ctx.fillStyle = 'white';
  ctx.font      = '14px monospace';
  ctx.fillText(`GPU Mem: ${gpuMB} MB`, 10, 40);
  ctx.fillText(`Tensors: ${mem.numTensors}`, 10, 56);
  if (performance.memory) {
    const heapMB = (performance.memory.usedJSHeapSize / 1024 / 1024).toFixed(2);
    ctx.fillText(`JS Heap: ${heapMB} MB`, 10, 72);
  }
  requestAnimationFrame(detectLoop);
}
