// tracker.js

export class Track {
  constructor(id, bbox, embedding, label = '', score = 0.0) {
    this.id = id;
    this.bbox = bbox;            // [y1, x1, y2, x2]
    this.embedding = DeepSort.smoothEmbedding(this.embedding, embedding, 0.8);
    this.age = 0;
    this.label = label;
    this.score = score;
  }

  predict() {
    return this.bbox;
  }

  update(bbox, embedding, label, score) {
    this.bbox = bbox;
    this.embedding = embedding;
    this.label = label;
    this.score = score;
    this.age = 0;
  }

}

export class DeepSort {
  constructor(maxAge = 30, maxCosineDistance = 0.3) {
    this.tracks = [];
    this.nextId = 1;
    this.maxAge = maxAge;
    this.maxDist = maxCosineDistance;
  }

  static iou(b1, b2) {
    const [y1, x1, y2, x2] = b1;
    const [Y1, X1, Y2, X2] = b2;
    const interW = Math.max(0, Math.min(x2, X2) - Math.max(x1, X1));
    const interH = Math.max(0, Math.min(y2, Y2) - Math.max(y1, Y1));
    const interArea = interW * interH;
    const area1 = (x2 - x1) * (y2 - y1);
    const area2 = (X2 - X1) * (Y2 - Y1);
    return interArea / (area1 + area2 - interArea);
  }

  static smoothEmbedding(prev, next, alpha = 0.8) {
    if (!prev) return next.slice(); // first assignment — no smoothing
    const result = new Array(prev.length);
    for (let i = 0; i < prev.length; i++) {
      result[i] = alpha * prev[i] + (1 - alpha) * next[i];
    }
   return result;
  }

  static cosineDist(e1, e2) {
    let dot = 0, n1 = 0, n2 = 0;
    for (let i = 0; i < e1.length; i++) {
      dot += e1[i] * e2[i];
      n1  += e1[i] * e1[i];
      n2  += e2[i] * e2[i];
    }
    return 1 - dot / Math.sqrt(n1 * n2);
  }

  async update(bboxes, embeddings, labels = [], scores = []) {
    const M = bboxes.length;
    const N = this.tracks.length;

    if (N === 0) {
      for (let j = 0; j < M; j++) {
        this.tracks.push(
          new Track(this.nextId++, bboxes[j], embeddings[j], labels[j], scores[j])
        );
      }
      return this.tracks;
    }

    this.tracks.forEach(t => t.predict());

    const cost = Array(N).fill().map(() => Array(M).fill(0));
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < M; j++) {
        const motion     = 1 - DeepSort.iou(this.tracks[i].bbox, bboxes[j]);
        const appearance = DeepSort.cosineDist(this.tracks[i].embedding, embeddings[j]);
        cost[i][j] = motion + appearance;
      }
    }

    const assignments = computeMunkres(cost);
    const assignedTracks = new Set();
    const assignedDetections = new Set();

    for (const [i, j] of assignments) {
      if (i < N && j < M && cost[i][j] < this.maxDist) {
        this.tracks[i].update(bboxes[j], embeddings[j], labels[j], scores[j]);
        assignedTracks.add(i);
        assignedDetections.add(j);
      }
    }

    this.tracks = this.tracks.filter((t, idx) => {
      if (!assignedTracks.has(idx)) t.age++;
      return t.age <= this.maxAge;
    });

    for (let j = 0; j < M; j++) {
      if (!assignedDetections.has(j)) {
        this.tracks.push(
          new Track(this.nextId++, bboxes[j], embeddings[j], labels[j], scores[j])
        );
      }
    }

    return this.tracks;
  }
}
