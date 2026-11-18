// 三角网格 → 障碍物体素掩码
// 1) 每个三角形与格子做 SAT 三角-盒重叠测试,标记表面格子
// 2) 从计算域边界洪泛填充外部区域,剩下的即内部 → 也是固体
// 3) 统计正面投影面积(Cd 参考面积)
import type { MeshData } from './carmesh';

export interface VoxelResult {
  mask: Uint8Array<ArrayBuffer>; // nx*ny*nz,1 = 固体(仅内部区域,不含边界层)
  area: number; // 正面投影面积(格子数)
}

// Akenine-Möller 三角-盒重叠测试(通用分离轴形式)
function triBoxOverlap(
  cx: number, cy: number, cz: number, h: number,
  t: number[], // 9 个数:v0 v1 v2
): boolean {
  const v = [
    t[0] - cx, t[1] - cy, t[2] - cz,
    t[3] - cx, t[4] - cy, t[5] - cz,
    t[6] - cx, t[7] - cy, t[8] - cz,
  ];
  const e = [
    v[3] - v[0], v[4] - v[1], v[5] - v[2],
    v[6] - v[3], v[7] - v[4], v[8] - v[5],
    v[0] - v[6], v[1] - v[7], v[2] - v[8],
  ];
  const sep = (ax: number, ay: number, az: number): boolean => {
    const p0 = ax * v[0] + ay * v[1] + az * v[2];
    const p1 = ax * v[3] + ay * v[4] + az * v[5];
    const p2 = ax * v[6] + ay * v[7] + az * v[8];
    const r = h * (Math.abs(ax) + Math.abs(ay) + Math.abs(az));
    const mn = Math.min(p0, p1, p2), mx = Math.max(p0, p1, p2);
    return mn > r || mx < -r;
  };
  // 9 个边叉积轴
  for (let k = 0; k < 3; k++) {
    const ex = e[k * 3], ey = e[k * 3 + 1], ez = e[k * 3 + 2];
    if (sep(0, -ez, ey)) return false;
    if (sep(ez, 0, -ex)) return false;
    if (sep(-ey, ex, 0)) return false;
  }
  // 3 个盒面法向轴(AABB 测试)
  for (let a = 0; a < 3; a++) {
    const mn = Math.min(v[a], v[3 + a], v[6 + a]);
    const mx = Math.max(v[a], v[3 + a], v[6 + a]);
    if (mn > h || mx < -h) return false;
  }
  // 三角平面法向轴
  const nx = e[1] * e[5] - e[2] * e[4];
  const ny = e[2] * e[3] - e[0] * e[5];
  const nz = e[0] * e[4] - e[1] * e[3];
  if (sep(nx, ny, nz)) return false;
  return true;
}

export function voxelize(mesh: MeshData, nx: number, ny: number, nz: number): VoxelResult {
  const nCells = nx * ny * nz;
  const mask = new Uint8Array(nCells);
  const pos = mesh.positions;
  const idx = mesh.indices;
  const tri = new Array<number>(9);

  for (let t = 0; t < idx.length; t += 3) {
    for (let v = 0; v < 3; v++) {
      const vi = idx[t + v] * 3;
      tri[v * 3] = pos[vi];
      tri[v * 3 + 1] = pos[vi + 1];
      tri[v * 3 + 2] = pos[vi + 2];
    }
    // 包围盒,限制在内部区域 [1, n-2]
    const lo = [0, 0, 0], hi = [0, 0, 0];
    for (let a = 0; a < 3; a++) {
      const mn = Math.min(tri[a], tri[3 + a], tri[6 + a]);
      const mx = Math.max(tri[a], tri[3 + a], tri[6 + a]);
      lo[a] = Math.max(1, Math.floor(mn - 0.5));
      hi[a] = Math.min(nx - 2, Math.ceil(mx + 0.5));
    }
    hi[1] = Math.min(ny - 2, hi[1]);
    hi[2] = Math.min(nz - 2, hi[2]);
    lo[1] = Math.max(1, lo[1]);
    lo[2] = Math.max(1, lo[2]);
    for (let z = lo[2]; z <= hi[2]; z++)
      for (let y = lo[1]; y <= hi[1]; y++)
        for (let x = lo[0]; x <= hi[0]; x++) {
          const id = x + y * nx + z * nx * ny;
          if (mask[id] === 0 && triBoxOverlap(x + 0.5, y + 0.5, z + 0.5, 0.5, tri)) mask[id] = 1;
        }
  }

  // 洪泛填充外部(6 连通),从所有边界面出发
  const stack: number[] = [];
  const pushIf = (id: number) => {
    if (mask[id] === 0) {
      mask[id] = 2;
      stack.push(id);
    }
  };
  for (let z = 0; z < nz; z++)
    for (let y = 0; y < ny; y++)
      for (const x of [0, nx - 1]) pushIf(x + y * nx + z * nx * ny);
  for (let z = 0; z < nz; z++)
    for (let x = 0; x < nx; x++)
      for (const y of [0, ny - 1]) pushIf(x + y * nx + z * nx * ny);
  for (let y = 0; y < ny; y++)
    for (let x = 0; x < nx; x++)
      for (const z of [0, nz - 1]) pushIf(x + y * nx + z * nx * ny);
  while (stack.length) {
    const id = stack.pop()!;
    const x = id % nx, y = ((id / nx) | 0) % ny, z = (id / (nx * ny)) | 0;
    if (x > 0) pushIf(id - 1);
    if (x < nx - 1) pushIf(id + 1);
    if (y > 0) pushIf(id - nx);
    if (y < ny - 1) pushIf(id + nx);
    if (z > 0) pushIf(id - nx * ny);
    if (z < nz - 1) pushIf(id + nx * ny);
  }

  // 内部(仍为 0)→ 固体;外部(2)→ 恢复流体
  for (let i = 0; i < nCells; i++) {
    if (mask[i] === 0) mask[i] = 1;
    else if (mask[i] === 2) mask[i] = 0;
  }

  // 正面投影面积(y-z 平面)
  let area = 0;
  for (let z = 0; z < nz; z++)
    for (let y = 0; y < ny; y++) {
      let solid = false;
      const base = y * nx + z * nx * ny;
      for (let x = 0; x < nx; x++)
        if (mask[base + x] === 1) {
          solid = true;
          break;
        }
      if (solid) area++;
    }
  return { mask, area };
}
