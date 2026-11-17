// 牙龈变形测试：边界无缝、边长度变化（撕裂指标）受控。
import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { buildDemoArch } from '../core/synthetic';
import { weldMesh } from '../core/stl';
import { segmentModel } from '../core/segmentation';
import { buildFrames } from '../core/tooth-frame';
import { buildGumBinding, deformGum } from '../core/gum-deform';
import { initialPoses, buildPlan, makeGoal, samplePoses } from '../core/planner';

function demo() {
  const built = buildDemoArch();
  const full = weldMesh(built.raw);
  const model = segmentModel(full);
  const frames = buildFrames(model.teeth);
  return { model, frames };
}

describe('牙龈跟随变形', () => {
  test('绑定建立成功且权重归一', () => {
    const { model } = demo();
    const binding = buildGumBinding(model);
    assert.ok(binding);
    const n = binding!.gumPart.vertexCount;
    for (let vi = 0; vi < n; vi++) {
      let sum = 0;
      for (let j = 0; j < binding!.toothOrder.length; j++) {
        const w: number = binding!.weights[vi * binding!.toothOrder.length + j];
        assert.ok(w >= 0, '权重不能为负');
        sum += w;
      }
      // 影响半径内归一为 1；半径外为零权重静止锚点
      assert.ok(Math.abs(sum - 1) < 1e-5 || sum === 0, `顶点 ${vi} 权重和=${sum}`);
    }
  });

  test('牙移动后牙龈边长度变化 <= 0.05mm（不撕裂）', () => {
    const { model, frames } = demo();
    const binding = buildGumBinding(model)!;
    const goals = new Map();
    // 让前牙做一个较大的平移 + 扭转
    const g1 = makeGoal(11);
    g1.translation = [0, 0, 1.5];
    g1.yawDeg = 8;
    goals.set(11, g1);
    const g2 = makeGoal(21);
    g2.translation = [0.5, 0.3, 0.8];
    goals.set(21, g2);
    const plan = buildPlan(frames, goals, { maxTranslationPerStep: 0.3, maxRotationPerStepDeg: 2 });

    let worstStrain = 0;
    for (let s = 0; s <= plan.steps.length; s++) {
      const poses = samplePoses(plan, frames, s);
      const deformed = deformGum(binding, model.teeth, frames, poses);
      // 用牙龈局部网格重建边应变：构造与 gumPart 同尺寸的网格度量
      const strain = edgeStrainOnPart(binding.gumPart.indices, binding.gumPart.positions, deformed);
      worstStrain = Math.max(worstStrain, strain);
    }
    assert.ok(worstStrain <= 0.05, `最大边长变化 ${worstStrain}mm 超过 0.05mm`);
  });

  test('无移动时牙龈位置不变', () => {
    const { model, frames } = demo();
    const binding = buildGumBinding(model)!;
    const poses = initialPoses(frames);
    const deformed = deformGum(binding, model.teeth, frames, poses);
    let maxDiff = 0;
    for (let i = 0; i < deformed.length; i++) {
      maxDiff = Math.max(maxDiff, Math.abs(deformed[i] - binding.gumPart.positions[i]));
    }
    assert.ok(maxDiff < 1e-5, `静止状态牙龈漂移 ${maxDiff}`);
  });
});

/** 在部件局部索引空间内计算最大边长变化 */
function edgeStrainOnPart(indices: Uint32Array, before: Float32Array, after: Float32Array): number {
  const seen = new Set<number>();
  const key = (a: number, b: number) => (a < b ? a * 1000003 + b : b * 1000003 + a);
  let worst = 0;
  for (let f = 0; f < indices.length; f += 3) {
    const tri = [indices[f], indices[f + 1], indices[f + 2]];
    for (const [a, b] of [[tri[0], tri[1]], [tri[1], tri[2]], [tri[2], tri[0]]] as const) {
      const k = key(a, b);
      if (seen.has(k)) continue;
      seen.add(k);
      const d0 = Math.hypot(
        before[a * 3] - before[b * 3],
        before[a * 3 + 1] - before[b * 3 + 1],
        before[a * 3 + 2] - before[b * 3 + 2],
      );
      const d1 = Math.hypot(
        after[a * 3] - after[b * 3],
        after[a * 3 + 1] - after[b * 3 + 1],
        after[a * 3 + 2] - after[b * 3 + 2],
      );
      worst = Math.max(worst, Math.abs(d1 - d0));
    }
  }
  return worst;
}
