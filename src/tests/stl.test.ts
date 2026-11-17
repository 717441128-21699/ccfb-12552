// STL 二进制往返与标签旁车文件解析测试。
import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { parseSTL, parseFaceLabels, weldMesh, computeSmoothNormals } from '../core/stl';
import { buildTwoToothTestModel } from '../core/synthetic';

function encodeBinarySTL(positions: Float32Array, indices: Uint32Array): ArrayBuffer {
  const triCount = indices.length / 3;
  const buf = new ArrayBuffer(84 + triCount * 50);
  const dv = new DataView(buf);
  dv.setUint32(80, triCount, true);
  let off = 84;
  for (let f = 0; f < triCount; f++) {
    // 法线置零
    dv.setFloat32(off, 0, true); dv.setFloat32(off + 4, 0, true); dv.setFloat32(off + 8, 0, true);
    off += 12;
    for (let k = 0; k < 3; k++) {
      const vi = indices[f * 3 + k];
      for (let c = 0; c < 3; c++) {
        dv.setFloat32(off, positions[vi * 3 + c], true);
        off += 4;
      }
    }
    dv.setUint16(off, 0, true);
    off += 2;
  }
  return buf;
}

describe('STL 解析与焊接', () => {
  test('二进制 STL 往返三角形数量与坐标一致', () => {
    const built = buildTwoToothTestModel();
    const buf = encodeBinarySTL(built.raw.positions, built.raw.indices);
    const parsed = parseSTL(buf);
    assert.equal(parsed.faceLabels.length, built.raw.faceLabels.length);
    assert.equal(parsed.positions.length, built.raw.indices.length * 3); // 面汤：每面 3 个独立顶点
    // STL 是面汤：按三角形展开后逐坐标比较
    for (let f = 0; f < built.raw.indices.length / 3; f++) {
      for (let k = 0; k < 3; k++) {
        const src = built.raw.indices[f * 3 + k];
        for (let c = 0; c < 3; c++) {
          const got = parsed.positions[(f * 3 + k) * 3 + c];
          const want = built.raw.positions[src * 3 + c];
          assert.ok(Math.abs(got - want) < 1e-6, `面${f}顶点${k}轴${c}`);
        }
      }
    }
  });

  test('带标签文件解析后逐面标签保留', () => {
    const built = buildTwoToothTestModel();
    const buf = encodeBinarySTL(built.raw.positions, built.raw.indices);
    const labelText = built.raw.faceLabels.join('\n');
    const labels = parseFaceLabels(labelText);
    const parsed = parseSTL(buf, labels);
    assert.deepEqual([...parsed.faceLabels], [...built.raw.faceLabels]);
  });

  test('JSON 标签格式兼容', () => {
    const labels = parseFaceLabels('{"faceLabels":[0,11,21,0]}');
    assert.deepEqual([...labels], [0, 11, 21, 0]);
  });

  test('焊接把共享坐标的重复顶点合并', () => {
    // 两个三角形共享一条边（4 个唯一顶点）
    const positions = Float32Array.from([
      0, 0, 0, 1, 0, 0, 1, 1, 0,
      0, 0, 0, 1, 1, 0, 0, 1, 0,
    ]);
    const indices = Uint32Array.from([0, 1, 2, 3, 4, 5]);
    const raw = { positions, indices, faceLabels: new Uint16Array([1, 1]) };
    const mesh = weldMesh(raw);
    assert.equal(mesh.vertexCount, 4);
    assert.equal(mesh.faceCount, 2);
    assert.equal(mesh.normals.length, 12);
  });

  test('法线计算为单位向量', () => {
    const positions = Float32Array.from([0, 0, 0, 1, 0, 0, 0, 1, 0]);
    const indices = Uint32Array.from([0, 1, 2]);
    const n = computeSmoothNormals(positions, indices);
    const l = Math.hypot(n[2], n[5], n[8]);
    // 三个顶点法线同向 +z
    assert.ok(Math.abs(n[2] - 1) < 1e-6);
    assert.ok(Math.abs(l - Math.sqrt(3)) < 1e-6);
  });
});
