// 验收测试：两颗已知间距测试牙，去釉量与几何真值偏差 <= 0.05mm。
import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { buildTwoToothTestModel } from '../core/synthetic';
import { weldMesh } from '../core/stl';
import { segmentModel } from '../core/segmentation';
import { buildFrames } from '../core/tooth-frame';
import { computeIPR } from '../core/ipr';

function setup(gap: number, width = 4) {
  const built = buildTwoToothTestModel(width, 4, 8, gap);
  const welded = weldMesh(built.raw);
  const model = segmentModel(welded);
  const frames = buildFrames(model.teeth);
  const report = computeIPR(model.teeth, frames, undefined, { maxPerSurface: 0.5 });
  return { built, model, frames, report };
}

describe('IPR 邻面去釉真值校验', () => {
  test('重叠 0.6mm：每面 0.3mm，误差 <= 0.05mm', () => {
    const gap = -0.6;
    const { built, report } = setup(gap);
    const truth = (built.meta as { truth: { overlap: number; stripPerSurface: number } }).truth;
    assert.equal(report.pairs.length, 1);
    const pair = report.pairs[0];

    assert.ok(Math.abs(pair.overlap - truth.overlap) <= 0.05,
      `重叠量 ${pair.overlap} 与真值 ${truth.overlap} 偏差超 0.05mm`);
    assert.ok(Math.abs(pair.stripA - truth.stripPerSurface) <= 0.05,
      `A 面去釉 ${pair.stripA} 与真值 ${truth.stripPerSurface} 偏差超 0.05mm`);
    assert.ok(Math.abs(pair.stripB - truth.stripPerSurface) <= 0.05,
      `B 面去釉 ${pair.stripB} 与真值 ${truth.stripPerSurface} 偏差超 0.05mm`);
    assert.ok(pair.stripA <= 0.5 + 1e-9 && pair.stripB <= 0.5 + 1e-9, '单面不得超过 0.5mm');
  });

  test('不同重叠量扫描：全部落在 0.05mm 容差内', () => {
    for (const gap of [-0.2, -0.35, -0.8, -1.0, -1.4]) {
      const { built, report } = setup(gap);
      const truth = (built.meta as { truth: { overlap: number; stripPerSurface: number } }).truth;
      const pair = report.pairs[0];
      assert.ok(Math.abs(pair.required - truth.overlap) <= 0.05,
        `gap=${gap}: required=${pair.required} truth=${truth.overlap}`);
      // 总量守恒（未被封顶时）
      if (truth.stripPerSurface <= 0.5) {
        assert.ok(Math.abs(pair.stripA + pair.stripB - truth.overlap) <= 0.05);
      }
    }
  });

  test('单面 0.5mm 封顶：重叠 2mm 时报告缺口', () => {
    const { report } = setup(-2.0);
    const pair = report.pairs[0];
    assert.ok(pair.stripA <= 0.5 + 1e-9);
    assert.ok(pair.stripB <= 0.5 + 1e-9);
    assert.ok(pair.shortfall > 0, '应报告无法满足的去釉缺口');
    assert.equal(report.capped, true);
  });

  test('存在间隙时不去釉', () => {
    const { report } = setup(0.8);
    assert.equal(report.pairs[0].required, 0);
    assert.equal(report.pairs[0].stripA, 0);
    assert.equal(report.pairs[0].stripB, 0);
  });

  test('牙冠宽度测量值等于真值', () => {
    const width = 5;
    const { report } = setup(-0.4, width);
    const pair = report.pairs[0];
    assert.ok(Math.abs(pair.crownWidthA - width) <= 0.05);
    assert.ok(Math.abs(pair.crownWidthB - width) <= 0.05);
  });

  test('分割后恰好得到两颗牙且标签正确', () => {
    const { model } = setup(-0.6);
    assert.equal(model.teeth.length, 2);
    assert.deepEqual(model.teeth.map((t) => t.label), [11, 21]);
  });
});
