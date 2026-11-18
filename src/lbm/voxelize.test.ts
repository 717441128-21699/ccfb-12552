/**
 * 体素化与 OBJ 解析的 CPU 单元测试。
 * 运行：npx tsx src/lbm/voxelize.test.ts
 */
import assert from 'node:assert/strict';
import { voxelizeMesh, type TriangleMesh } from './voxelize';
import { parseOBJ, buildCarMesh, MeshBuilder } from './carMesh';
import { frontalArea } from './aerodynamics';
import { SOLID } from './constants';

let passed = 0;
const check = (name: string, fn: () => void) => { fn(); passed++; console.log(`  ✓ ${name}`); };

// ---- 测试 1：单个轴对齐立方体应体素化成实心块 ----
{
  const mb = new MeshBuilder();
  mb.addBox(0, 0, 0, 1, 1, 1);
  const mesh = mb.build();
  const nx = 40, ny = 30, nz = 30;
  const vox = voxelizeMesh(mesh, { nx, ny, nz });

  let minX = nx, maxX = -1, minY = ny, maxY = -1, minZ = nz, maxZ = -1;
  const id = (i: number, j: number, k: number) => i * (ny * nz) + j * nz + k;
  for (let i = 0; i < nx; i++) for (let j = 0; j < ny; j++) for (let k = 0; k < nz; k++) {
    if (vox.flags[id(i, j, k)] === SOLID) {
      minX = Math.min(minX, i); maxX = Math.max(maxX, i);
      minY = Math.min(minY, j); maxY = Math.max(maxY, j);
      minZ = Math.min(minZ, k); maxZ = Math.max(maxZ, k);
    }
  }
  check('立方体产生固体格', () => assert.ok(vox.solidCount > 0));

  // 内部应为实心：取包围盒中心列，沿 x 方向所有格都固体
  const cy = (minY + maxY) >> 1, cz = (minZ + maxZ) >> 1;
  let interiorHoles = 0;
  for (let i = minX + 1; i < maxX; i++) {
    if (vox.flags[id(i, cy, cz)] !== SOLID) interiorHoles++;
  }
  check('实心立方体内部无空腔', () => assert.equal(interiorHoles, 0));
  check('固体包围盒三个维度都非平凡', () =>
    assert.ok(maxX - minX > 1 && maxY - minY > 1 && maxZ - minZ > 1));
}

// ---- 测试 2：参数车两个造型的体素结果确实不同（后视镜/尾翼）----
{
  const g = { nx: 120, ny: 80, nz: 80 };
  const a = voxelizeMesh(
    buildCarMesh({ mirrorOffset: 0, wingAngleDeg: 0, wingHeight: 0 }), g);
  const b = voxelizeMesh(
    buildCarMesh({ mirrorOffset: 8, wingAngleDeg: 25, wingHeight: 6 }), g);
  let diff = 0;
  for (let i = 0; i < a.flags.length; i++) if (a.flags[i] !== b.flags[i]) diff++;
  check('不同造型参数产生不同体素栅格', () => assert.ok(diff > 50, `diff=${diff}`));
  check('两种造型固体量同量级（未塌缩）', () =>
    assert.ok(Math.abs(a.solidCount - b.solidCount) / a.solidCount < 0.25));
}

// ---- 测试 3：迎风投影面积不超过 yz 截面且为正 ----
{
  const g = { nx: 120, ny: 80, nz: 80 };
  const vox = voxelizeMesh(buildCarMesh({ mirrorOffset: 2, wingAngleDeg: 8, wingHeight: 2 }), g);
  const area = frontalArea(vox.flags, g.nx, g.ny, g.nz);
  check('迎风面积为正', () => assert.ok(area > 0));
  check('迎风面积不超过 yz 截面', () => assert.ok(area <= g.ny * g.nz));
  check('迎风面积小于固体总数', () => assert.ok(area < vox.solidCount));
}

// ---- 测试 4：OBJ 解析（三角形 + 四边形扇面 + 负索引）----
{
  const obj = [
    'v 0 0 0', 'v 1 0 0', 'v 1 1 0', 'v 0 1 0',
    'v 0 0 1', 'v 1 0 1', 'v 1 1 1', 'v 0 1 1',
    'f 1 2 3 4',       // 底面
    'f -4 -3 -2 -1',   // 顶面（负索引）
    'f 1 2 6 5',       // 前面
    'f 2 3 7 6',       // 右面
    'f 3 4 8 7',       // 后面
    'f 4 1 5 8',       // 左面
  ].join('\n');
  const mesh: TriangleMesh = parseOBJ(obj);
  check('OBJ 六个四边形被三角化（12 个三角形 = 36 顶点）', () =>
    assert.equal(mesh.positions.length / 3, 36));
  check('OBJ 负索引正确解析（首顶点坐标）', () => {
    // 第二个面的第一个顶点是 -4 → 第 5 个顶点 (0,0,1)
    const base = 9 * 3; // 前 3 个顶点（第一个三角形）之后
    assert.equal(mesh.positions[base], 0);
    assert.equal(mesh.positions[base + 2], 1);
  });

  // 解析结果能正常体素化
  const vox = voxelizeMesh(mesh, { nx: 32, ny: 32, nz: 32 });
  check('OBJ 网格可体素化', () => assert.ok(vox.solidCount > 0));
}

// ---- 测试 5：体素化对平移/尺度不敏感（包围盒归一化）----
{
  const mb1 = new MeshBuilder();
  mb1.addBox(0, 0, 0, 1, 1, 1);
  const small = voxelizeMesh(mb1.build(), { nx: 50, ny: 40, nz: 40 });

  const mb2 = new MeshBuilder();
  mb2.addBox(100, -50, 30, 10, 10, 10); // 大幅平移 + 放大
  const big = voxelizeMesh(mb2.build(), { nx: 50, ny: 40, nz: 40 });
  // 归一化后两个单位立方体应占相近的格子数
  check('包围盒归一化使不同尺度模型占用相近', () => {
    const ratio = Math.min(small.solidCount, big.solidCount) /
      Math.max(small.solidCount, big.solidCount);
    assert.ok(ratio > 0.5, `ratio=${ratio}`);
  });
}

console.log(`\n全部 ${passed} 项体素化测试通过 ✔`);
