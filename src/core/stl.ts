// STL 二进制 / ASCII 解析，以及顶点焊接与面标签旁车文件读取。
import type { Mesh, RawMesh } from './mesh';

/** 解析 STL（自动识别二进制与 ASCII）。无标签时所有面标签为 0。 */
export function parseSTL(buffer: ArrayBuffer, faceLabels?: Uint16Array): RawMesh {
  const bytes = new Uint8Array(buffer);
  const isAscii = looksAscii(bytes);
  if (isAscii) return parseAsciiSTL(new TextDecoder().decode(bytes), faceLabels);
  return parseBinarySTL(bytes, faceLabels);
}

function looksAscii(bytes: Uint8Array): boolean {
  // ASCII STL 以 "solid" 开头，且在文件头 512 字节内出现 "facet"
  const head = new TextDecoder().decode(bytes.subarray(0, Math.min(512, bytes.length)));
  return /^\s*solid/i.test(head) && /facet/i.test(head);
}

function parseBinarySTL(bytes: Uint8Array, labels?: Uint16Array): RawMesh {
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const triCount = dv.getUint32(80, true);
  const expected = 84 + triCount * 50;
  if (expected > bytes.byteLength) {
    throw new Error(`STL 数据不完整：声明 ${triCount} 个三角形，需要 ${expected} 字节，实际 ${bytes.byteLength}`);
  }
  const positions = new Float32Array(triCount * 9);
  const indices = new Uint32Array(triCount * 3);
  const faceLabels = new Uint16Array(triCount);
  let offset = 84;
  for (let i = 0; i < triCount; i++) {
    offset += 12; // 跳过面法线
    const base = i * 9;
    for (let k = 0; k < 9; k++) {
      positions[base + k] = dv.getFloat32(offset, true);
      offset += 4;
    }
    offset += 2; // 属性字节
    indices[i * 3] = i * 3;
    indices[i * 3 + 1] = i * 3 + 1;
    indices[i * 3 + 2] = i * 3 + 2;
    faceLabels[i] = labels && i < labels.length ? labels[i] : 0;
  }
  return { positions, indices, faceLabels };
}

function parseAsciiSTL(text: string, labels?: Uint16Array): RawMesh {
  const floats = text.match(/-?\d*\.?\d+(?:[eE][+-]?\d+)?/g) ?? [];
  // 每个 facet：3 法线 + 9 顶点 = 12 个数
  const perTri = 12;
  if (floats.length % perTri !== 0) {
    throw new Error('ASCII STL 数值数量异常，无法解析');
  }
  const triCount = floats.length / perTri;
  const all = floats.map(Number);
  const positions = new Float32Array(triCount * 9);
  const indices = new Uint32Array(triCount * 3);
  const faceLabels = new Uint16Array(triCount);
  for (let i = 0; i < triCount; i++) {
    const src = i * perTri + 3;
    const dst = i * 9;
    for (let k = 0; k < 9; k++) positions[dst + k] = all[src + k];
    indices[i * 3] = i * 3;
    indices[i * 3 + 1] = i * 3 + 1;
    indices[i * 3 + 2] = i * 3 + 2;
    faceLabels[i] = labels && i < labels.length ? labels[i] : 0;
  }
  return { positions, indices, faceLabels };
}

/**
 * 解析旁车标签文件，支持两种格式：
 *  1. 纯文本：每行一个整数（按三角形序号），# 开头为注释
 *  2. JSON：{ "faceLabels": [..] } 或直接是数组
 */
export function parseFaceLabels(text: string): Uint16Array {
  const trimmed = text.trim();
  if (trimmed.startsWith('{') || trimmed.startsWith('[')) {
    const obj = JSON.parse(trimmed);
    const arr: number[] = Array.isArray(obj) ? obj : obj.faceLabels;
    if (!Array.isArray(arr)) throw new Error('标签 JSON 中找不到 faceLabels 数组');
    return Uint16Array.from(arr);
  }
  const nums: number[] = [];
  for (const line of text.split(/\r?\n/)) {
    const s = line.trim();
    if (!s || s.startsWith('#')) continue;
    const n = Number.parseInt(s, 10);
    if (Number.isFinite(n)) nums.push(n);
  }
  return Uint16Array.from(nums);
}

/**
 * 顶点焊接：量化空间哈希，把距离 <= eps 的重复顶点合并。
 * 口扫 STL 通常每三角形独立顶点，必须焊接后牙龈变形才不会撕裂。
 */
export function weldMesh(raw: RawMesh, eps = 1e-4): Mesh {
  const srcCount = raw.positions.length / 3;
  const inv = 1 / eps;
  // 量化坐标 -> 焊接顶点号。量化格内坐标完全相同（STL 共享顶点本就逐位相等），
  // 用分桶 + 桶内精确比较，避免整数乘法哈希的碰撞与溢出问题。
  const buckets = new Map<string, number[]>();
  const weldMap = new Uint32Array(srcCount);
  const kept: number[] = [];

  const quantKey = (ix: number, iy: number, iz: number): string => `${ix}|${iy}|${iz}`;

  for (let i = 0; i < srcCount; i++) {
    const x = raw.positions[i * 3];
    const y = raw.positions[i * 3 + 1];
    const z = raw.positions[i * 3 + 2];
    const ix = Math.round(x * inv);
    const iy = Math.round(y * inv);
    const iz = Math.round(z * inv);
    const key = quantKey(ix, iy, iz);
    const bucket = buckets.get(key);
    let id: number | undefined;
    if (bucket) {
      // 桶内逐坐标精确匹配（同格理论上只有一个焊接点，这里防御浮点边界）
      for (const cand of bucket) {
        if (
          kept[cand * 3] === x &&
          kept[cand * 3 + 1] === y &&
          kept[cand * 3 + 2] === z
        ) { id = cand; break; }
      }
    }
    if (id === undefined) {
      id = kept.length / 3;
      const arr = bucket ?? [];
      arr.push(id);
      buckets.set(key, arr);
      weldMap[i] = id;
      kept.push(x, y, z);
    } else {
      weldMap[i] = id;
    }
  }

  const vertexCount = kept.length / 3;
  const positions = new Float32Array(kept);
  const indices = new Uint32Array(raw.indices.length);
  for (let i = 0; i < raw.indices.length; i++) {
    indices[i] = weldMap[raw.indices[i]];
  }
  const normals = computeSmoothNormals(positions, indices);
  return {
    positions,
    normals,
    indices,
    faceLabels: raw.faceLabels,
    vertexCount,
    faceCount: raw.faceLabels.length,
    weldMap,
  };
}

/** 面积加权平滑法线；退化三角形安全跳过 */
export function computeSmoothNormals(positions: Float32Array, indices: Uint32Array): Float32Array {
  const normals = new Float32Array(positions.length);
  for (let f = 0; f < indices.length; f += 3) {
    const a = indices[f] * 3;
    const b = indices[f + 1] * 3;
    const c = indices[f + 2] * 3;
    const ux = positions[b] - positions[a];
    const uy = positions[b + 1] - positions[a + 1];
    const uz = positions[b + 2] - positions[a + 2];
    const vx = positions[c] - positions[a];
    const vy = positions[c + 1] - positions[a + 1];
    const vz = positions[c + 2] - positions[a + 2];
    const nx = uy * vz - uz * vy;
    const ny = uz * vx - ux * vz;
    const nz = ux * vy - uy * vx;
    normals[a] += nx; normals[a + 1] += ny; normals[a + 2] += nz;
    normals[b] += nx; normals[b + 1] += ny; normals[b + 2] += nz;
    normals[c] += nx; normals[c + 1] += ny; normals[c + 2] += nz;
  }
  for (let i = 0; i < normals.length; i += 3) {
    const l = Math.hypot(normals[i], normals[i + 1], normals[i + 2]);
    if (l > 1e-12) {
      normals[i] /= l; normals[i + 1] /= l; normals[i + 2] /= l;
    } else {
      normals[i + 2] = 1; // 退化点兜底
    }
  }
  return normals;
}

/** 从焊接全网格按面标签提取一个部件（顶点重新紧凑编号） */
export function extractPart(mesh: Mesh, label: number): import('./mesh').ToothPart {
  const usedVerts = new Set<number>();
  const faceCount = (() => {
    let n = 0;
    for (let f = 0; f < mesh.faceCount; f++) {
      if (mesh.faceLabels[f] === label) n++;
    }
    return n;
  })();
  const indices = new Uint32Array(faceCount * 3);
  let w = 0;
  for (let f = 0; f < mesh.faceCount; f++) {
    if (mesh.faceLabels[f] !== label) continue;
    for (let k = 0; k < 3; k++) {
      const vi = mesh.indices[f * 3 + k];
      usedVerts.add(vi);
      indices[w++] = vi;
    }
  }
  const remap = new Map<number, number>();
  const compact: number[] = [];
  let id = 0;
  for (const vi of usedVerts) {
    remap.set(vi, id++);
    compact.push(
      mesh.positions[vi * 3],
      mesh.positions[vi * 3 + 1],
      mesh.positions[vi * 3 + 2],
    );
  }
  for (let i = 0; i < indices.length; i++) indices[i] = remap.get(indices[i])!;
  const positions = new Float32Array(compact);
  const normals = computeSmoothNormals(positions, indices);
  return {
    label,
    name: label === 0 ? '牙龈' : `#${label}`,
    positions,
    normals,
    indices,
    vertexCount: id,
    faceCount,
  };
}
