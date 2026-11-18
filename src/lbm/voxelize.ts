/**
 * 规则网格体素化：把三角网格写入 Uint8 占用栅格。
 * 采用三方向射线求交 + 多数投票的实体判定，对任意朝向的闭合（水密）网格
 * 都能稳健得到实心体；单方向射线会漏掉与射线垂直的面，三方向投票消除该盲区。
 */
import { SOLID } from './constants';

export interface VoxelGrid {
  nx: number;
  ny: number;
  nz: number;
  /** 长度 nx*ny*nz，FLUID(0)/SOLID(1) */
  flags: Uint8Array;
  solidCount: number;
}

export interface TriangleMesh {
  positions: Float32Array; // 扁平 xyz
  indices?: Uint32Array; // 缺省则 positions 自身按三角形解释
}

export interface VoxelOptions {
  nx: number;
  ny: number;
  nz: number;
  /** 模型包围盒映射到格子空间后的边距（格） */
  margin?: number;
}

interface Tri {
  ax: number; ay: number; az: number;
  bx: number; by: number; bz: number;
  cx: number; cy: number; cz: number;
}

type Axis = 'x' | 'y' | 'z';

export function voxelizeMesh(mesh: TriangleMesh, opts: VoxelOptions): VoxelGrid {
  const { nx, ny, nz } = opts;
  const margin = opts.margin ?? Math.max(4, Math.round(Math.min(nx, ny, nz) * 0.06));
  const flags = new Uint8Array(nx * ny * nz);

  // 1. 模型包围盒 → 格子空间的统一缩放（保持比例），车底留离地间隙
  const pos = mesh.positions;
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  for (let i = 0; i < pos.length; i += 3) {
    const x = pos[i], y = pos[i + 1], z = pos[i + 2];
    if (x < minX) minX = x;
    if (y < minY) minY = y;
    if (z < minZ) minZ = z;
    if (x > maxX) maxX = x;
    if (y > maxY) maxY = y;
    if (z > maxZ) maxZ = z;
  }
  const sizeX = Math.max(maxX - minX, 1e-9);
  const sizeY = Math.max(maxY - minY, 1e-9);
  const sizeZ = Math.max(maxZ - minZ, 1e-9);

  const targetLenX = nx * 0.55;
  const s = Math.min(
    targetLenX / sizeX,
    (ny - 2 * margin) / sizeY,
    (nz - 2 * margin) / sizeZ,
  );
  const groundGap = Math.round(ny * 0.08);
  const offX = Math.round(nx * 0.18) - minX * s;
  const offY = margin + groundGap - minY * s;
  const offZ = nz / 2 - ((minZ + maxZ) / 2) * s;

  const mapX = (x: number) => x * s + offX;
  const mapY = (y: number) => y * s + offY;
  const mapZ = (z: number) => z * s + offZ;

  const idx = mesh.indices;
  const triCount = idx ? idx.length / 3 : pos.length / 9;
  const tris: Tri[] = [];
  for (let t = 0; t < triCount; t++) {
    const a = idx ? idx[t * 3] * 3 : t * 9;
    const b = idx ? idx[t * 3 + 1] * 3 : t * 9 + 3;
    const c = idx ? idx[t * 3 + 2] * 3 : t * 9 + 6;
    tris.push({
      ax: mapX(pos[a]), ay: mapY(pos[a + 1]), az: mapZ(pos[a + 2]),
      bx: mapX(pos[b]), by: mapY(pos[b + 1]), bz: mapZ(pos[b + 2]),
      cx: mapX(pos[c]), cy: mapY(pos[c + 1]), cz: mapZ(pos[c + 2]),
    });
  }

  // 2. 三方向射线多数投票
  const voteX = castAxis(tris, nx, ny, nz, 'x');
  const voteY = castAxis(tris, nx, ny, nz, 'y');
  const voteZ = castAxis(tris, nx, ny, nz, 'z');
  const strideX = ny * nz, strideY = nz;
  let solidCount = 0;
  for (let i = 0; i < nx; i++) {
    for (let j = 0; j < ny; j++) {
      const base = i * strideX + j * strideY;
      for (let k = 0; k < nz; k++) {
        const n = base + k;
        if (voteX[n] + voteY[n] + voteZ[n] >= 2) {
          flags[n] = SOLID;
          solidCount++;
        }
      }
    }
  }
  return { nx, ny, nz, flags, solidCount };
}

/**
 * 沿指定轴对所有格子中心射线做三角形求交 + 奇偶填充，
 * 返回 0/1 的内外投票数组。
 */
function castAxis(tris: Tri[], nx: number, ny: number, nz: number, axis: Axis): Uint8Array {
  const dims: Record<Axis, number> = { x: nx, y: ny, z: nz };
  const inside = new Uint8Array(nx * ny * nz);
  const strideX = ny * nz, strideY = nz;
  const others = (['x', 'y', 'z'].filter((a) => a !== axis)) as [Axis, Axis];
  const sizeU = dims[others[0]];
  const sizeV = dims[others[1]];

  for (let u = 0; u < sizeU; u++) {
    for (let v = 0; v < sizeV; v++) {
      const pu = u + 0.5, pv = v + 0.5;
      const hits: number[] = [];
      for (const t of tris) {
        const h = rayTriAxis(axis, pu, pv, t);
        if (h !== null && h >= 0 && h <= dims[axis]) hits.push(h);
      }
      hits.sort((a, b) => a - b);
      for (let q = 0; q + 1 < hits.length; q += 2) {
        const a0 = Math.max(0, Math.floor(hits[q]));
        const a1 = Math.min(dims[axis] - 1, Math.ceil(hits[q + 1]));
        for (let w = a0; w <= a1; w++) {
          let i: number, j: number, k: number;
          if (axis === 'x') { i = w; j = u; k = v; }
          else if (axis === 'y') { i = u; j = w; k = v; }
          else { i = u; j = v; k = w; }
          inside[i * strideX + j * strideY + k] = 1;
        }
      }
    }
  }
  return inside;
}

/**
 * 沿指定轴正方向、横截面坐标 (pu,pv) 的射线与三角形求交，
 * 返回交点在该轴上的坐标；不相交返回 null。
 * 坐标置换后等价于：射线 O=(0,pu,pv)、D=(1,0,0) 的标准 Möller–Trumbore。
 */
function rayTriAxis(axis: Axis, pu: number, pv: number, t: Tri): number | null {
  const perm = (x: number, y: number, z: number): [number, number, number] => {
    if (axis === 'x') return [x, y, z];
    if (axis === 'y') return [y, z, x];
    return [z, x, y];
  };
  const A = perm(t.ax, t.ay, t.az);
  const B = perm(t.bx, t.by, t.bz);
  const C = perm(t.cx, t.cy, t.cz);
  const E1: [number, number, number] = [B[0] - A[0], B[1] - A[1], B[2] - A[2]];
  const E2: [number, number, number] = [C[0] - A[0], C[1] - A[1], C[2] - A[2]];
  const D: [number, number, number] = [1, 0, 0];
  // h = D × E2
  const h: [number, number, number] = [
    D[1] * E2[2] - D[2] * E2[1],
    D[2] * E2[0] - D[0] * E2[2],
    D[0] * E2[1] - D[1] * E2[0],
  ];
  const det = E1[0] * h[0] + E1[1] * h[1] + E1[2] * h[2];
  if (Math.abs(det) < 1e-9) return null;
  const inv = 1 / det;
  const O: [number, number, number] = [0, pu, pv];
  const Tv: [number, number, number] = [O[0] - A[0], O[1] - A[1], O[2] - A[2]];
  const u = (Tv[0] * h[0] + Tv[1] * h[1] + Tv[2] * h[2]) * inv;
  if (u < 0 || u > 1) return null;
  // q = T × E1
  const q: [number, number, number] = [
    Tv[1] * E1[2] - Tv[2] * E1[1],
    Tv[2] * E1[0] - Tv[0] * E1[2],
    Tv[0] * E1[1] - Tv[1] * E1[0],
  ];
  const v = (D[0] * q[0] + D[1] * q[1] + D[2] * q[2]) * inv;
  if (v < 0 || u + v > 1) return null;
  return (E2[0] * q[0] + E2[1] * q[1] + E2[2] * q[2]) * inv;
}
