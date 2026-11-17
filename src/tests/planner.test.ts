// 规划拆步测试：每步平移/旋转不超上限，且超限时自动增加步数。
import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { buildTwoToothTestModel } from '../core/synthetic';
import { weldMesh } from '../core/stl';
import { segmentModel } from '../core/segmentation';
import { buildFrames } from '../core/tooth-frame';
import { buildPlan, initialPoses, makeGoal, poseDelta, samplePoses } from '../core/planner';

function framesOf() {
  const built = buildTwoToothTestModel(4, 4, 8, -0.6);
  const model = segmentModel(weldMesh(built.raw));
  return { model, frames: buildFrames(model.teeth) };
}

describe('移动规划与自动拆步', () => {
  test('平移 1.2mm、上限 0.3mm -> 至少 4 步且每步不超限', () => {
    const { frames } = framesOf();
    const goals = new Map();
    const g = makeGoal(11);
    g.translation = [1.2, 0, 0];
    goals.set(11, g);
    const plan = buildPlan(frames, goals, { maxTranslationPerStep: 0.3, maxRotationPerStepDeg: 2 });

    assert.ok(plan.steps.length >= 4, `应拆成 >=4 步，实际 ${plan.steps.length}`);
    for (const step of plan.steps) {
      const t = step.stepTranslation.get(11) ?? 0;
      assert.ok(t <= 0.3 + 1e-6, `单步平移 ${t} 超过 0.3mm`);
    }
    // 终态到位
    const finalPose = plan.steps[plan.steps.length - 1].poses.get(11)!;
    const d = poseDelta(initialPoses(frames).get(11)!, finalPose);
    assert.ok(Math.abs(d.translation - 1.2) <= 1e-6);
  });

  test('旋转 10°、上限 2° -> 5 步', () => {
    const { frames } = framesOf();
    const goals = new Map();
    const g = makeGoal(21);
    g.yawDeg = 10;
    goals.set(21, g);
    const plan = buildPlan(frames, goals, { maxTranslationPerStep: 0.3, maxRotationPerStepDeg: 2 });
    assert.equal(plan.steps.length, 5);
    for (const step of plan.steps) {
      const r = step.stepRotation.get(21) ?? 0;
      assert.ok(r <= 2 + 1e-6, `单步旋转 ${r} 超过 2°`);
    }
  });

  test('无规划时单步空方案，采样仍返回初始位姿', () => {
    const { frames } = framesOf();
    const plan = buildPlan(frames, new Map(), { maxTranslationPerStep: 0.3, maxRotationPerStepDeg: 2 });
    assert.equal(plan.steps.length, 1);
    const poses = samplePoses(plan, frames, 0.5);
    assert.ok(poses.has(11));
  });

  test('连续采样在两步之间给出插值位姿', () => {
    const { frames } = framesOf();
    const goals = new Map();
    const g = makeGoal(11);
    g.translation = [1.0, 0, 0];
    goals.set(11, g);
    const plan = buildPlan(frames, goals, { maxTranslationPerStep: 0.5, maxRotationPerStepDeg: 2 });
    const half = samplePoses(plan, frames, 1.5).get(11)!;
    const full = samplePoses(plan, frames, 2).get(11)!;
    const start = initialPoses(frames).get(11)!;
    const movedHalf = Math.hypot(...half.position.map((c, i) => c - start.position[i]) as [number, number, number]);
    const movedFull = Math.hypot(...full.position.map((c, i) => c - start.position[i]) as [number, number, number]);
    assert.ok(Math.abs(movedHalf - 0.75) < 1e-6, `半程位移 ${movedHalf}`);
    assert.ok(Math.abs(movedFull - 1.0) < 1e-6);
  });

  test('每步 activeLabels 只包含真正在动的牙', () => {
    const { frames } = framesOf();
    const goals = new Map();
    const g = makeGoal(11);
    g.translation = [0.6, 0, 0];
    goals.set(11, g);
    const plan = buildPlan(frames, goals, { maxTranslationPerStep: 0.3, maxRotationPerStepDeg: 2 });
    for (const step of plan.steps) {
      assert.deepEqual(step.activeLabels, [11]);
    }
  });
});
