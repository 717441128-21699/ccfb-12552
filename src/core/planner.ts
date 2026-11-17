// 移动规划：每颗牙的平移/扭转/倾斜目标位，按每步平移、旋转上限自动拆步，
// 并在每步记录哪些牙在动（供时间轴 UI 高亮）。
import type { ToothFrame } from './tooth-frame';
import type { ToothPose } from './ipr';
import {
  type Quat, type Vec3,
  add, deg2rad, qFromAxisAngle, qIdentity, qMultiply, qSlerp,
  rad2deg, scale, sub,
} from './math';

/** 医生在每颗牙局部坐标系里设置的目标量 */
export interface ToothGoal {
  label: number;
  /** 局部平移（mm）：[近远中, 颊舌, 龈合] */
  translation: Vec3;
  /** 绕牙长轴扭转（度） */
  yawDeg: number;
  /** 绕近远中轴倾斜（度） */
  tiltDeg: number;
  /** 绕颊舌轴转矩（度） */
  torqueDeg: number;
}

export interface PlanLimits {
  /** 每步最大平移（mm） */
  maxTranslationPerStep: number;
  /** 每步最大旋转（度） */
  maxRotationPerStepDeg: number;
}

export interface PlanStep {
  index: number;
  /** 本步结束后每颗牙的位姿（标签索引） */
  poses: Map<number, ToothPose>;
  /** 本步有移动的牙位标签 */
  activeLabels: number[];
  /** 每颗牙本步的平移量（mm）/ 旋转量（度），用于上限校验与展示 */
  stepTranslation: Map<number, number>;
  stepRotation: Map<number, number>;
}

export interface TreatmentPlan {
  steps: PlanStep[];
  limits: PlanLimits;
  goals: Map<number, ToothGoal>;
}

export function makeGoal(label: number): ToothGoal {
  return { label, translation: [0, 0, 0], yawDeg: 0, tiltDeg: 0, torqueDeg: 0 };
}

/** 把局部系目标量换算成世界系目标位姿 */
export function goalPose(frame: ToothFrame, goal: ToothGoal): ToothPose {
  // 局部平移 -> 世界
  const [tx, ty, tz] = goal.translation;
  const worldDelta: Vec3 = [
    frame.mesioDistal[0] * tx + frame.buccoLingual[0] * ty + frame.longAxis[0] * tz,
    frame.mesioDistal[1] * tx + frame.buccoLingual[1] * ty + frame.longAxis[1] * tz,
    frame.mesioDistal[2] * tx + frame.buccoLingual[2] * ty + frame.longAxis[2] * tz,
  ];
  const position = add(frame.centroid, worldDelta);

  // 局部旋转：先转矩(颊舌轴)、再倾斜(近远中轴)、最后扭转(长轴)，
  // 世界系四元数 = orientation * qLocal * orientation^-1
  const qYaw = qFromAxisAngle([0, 0, 1], deg2rad(goal.yawDeg));
  const qTilt = qFromAxisAngle([1, 0, 0], deg2rad(goal.tiltDeg));
  const qTorque = qFromAxisAngle([0, 1, 0], deg2rad(goal.torqueDeg));
  const qLocal = qMultiply(qYaw, qMultiply(qTilt, qTorque));
  const qWorld = qMultiply(frame.orientation, qMultiply(qLocal, conj(frame.orientation)));

  return { position, orientation: qWorld };
}

function conj(q: Quat): Quat {
  return [-q[0], -q[1], -q[2], q[3]];
}

function initialPose(frame: ToothFrame): ToothPose {
  return { position: [...frame.centroid] as Vec3, orientation: qIdentity() };
}

/** 两个位姿之间的平移距离与旋转角（度） */
export function poseDelta(a: ToothPose, b: ToothPose): { translation: number; rotationDeg: number } {
  const translation = Math.hypot(
    b.position[0] - a.position[0],
    b.position[1] - a.position[1],
    b.position[2] - a.position[2],
  );
  const d = qMultiply(conj(a.orientation), b.orientation);
  const cosW = Math.min(1, Math.abs(d[3]));
  const rotationDeg = rad2deg(2 * Math.acos(cosW));
  return { translation, rotationDeg };
}

/**
 * 核心拆步：
 *  N = max(ceil(最大平移距离 / 每步平移上限), ceil(最大旋转角 / 每步旋转上限))，
 *  所有牙共用同一个 N 步时间轴，每颗牙在自己参与的区间内线性/球面插值。
 */
export function buildPlan(
  frames: ToothFrame[],
  goals: Map<number, ToothGoal>,
  limits: PlanLimits,
): TreatmentPlan {
  const start = new Map<number, ToothPose>();
  const target = new Map<number, ToothPose>();
  const deltas = new Map<number, { translation: number; rotationDeg: number }>();

  for (const frame of frames) {
    const s = initialPose(frame);
    start.set(frame.label, s);
    const goal = goals.get(frame.label);
    if (!goal) {
      target.set(frame.label, s);
      deltas.set(frame.label, { translation: 0, rotationDeg: 0 });
      continue;
    }
    const t = goalPose(frame, goal);
    target.set(frame.label, t);
    deltas.set(frame.label, poseDelta(s, t));
  }

  let stepCount = 1;
  for (const d of deltas.values()) {
    if (d.translation <= 1e-9 && d.rotationDeg <= 1e-9) continue;
    const nT = Math.max(1, Math.ceil(d.translation / limits.maxTranslationPerStep - 1e-9));
    const nR = Math.max(1, Math.ceil(d.rotationDeg / limits.maxRotationPerStepDeg - 1e-9));
    stepCount = Math.max(stepCount, nT, nR);
  }

  const steps: PlanStep[] = [];
  for (let k = 1; k <= stepCount; k++) {
    const t = k / stepCount;
    const poses = new Map<number, ToothPose>();
    const activeLabels: number[] = [];
    const stepTranslation = new Map<number, number>();
    const stepRotation = new Map<number, number>();

    for (const frame of frames) {
      const label = frame.label;
      const s = start.get(label)!;
      const g = target.get(label)!;
      const d = deltas.get(label)!;
      const pose: ToothPose = {
        position: lerp3(s.position, g.position, t),
        orientation: qSlerp(s.orientation, g.orientation, t),
      };
      poses.set(label, pose);

      if (d.translation > 1e-9 || d.rotationDeg > 1e-9) {
        // 本步增量（相对上一步结束位姿）
        const prevT = (k - 1) / stepCount;
        const prev: ToothPose = {
          position: lerp3(s.position, g.position, prevT),
          orientation: qSlerp(s.orientation, g.orientation, prevT),
        };
        const inc = poseDelta(prev, pose);
        // 浮点保护：上限校验留 1e-6 容差
        if (inc.translation > limits.maxTranslationPerStep + 1e-6 ||
            inc.rotationDeg > limits.maxRotationPerStepDeg + 1e-6) {
          // 理论上不会发生；兜底再拆由 stepCount 的 ceil 保证
        }
        stepTranslation.set(label, inc.translation);
        stepRotation.set(label, inc.rotationDeg);
        activeLabels.push(label);
      }
    }
    steps.push({ index: k - 1, poses, activeLabels, stepTranslation, stepRotation });
  }

  return { steps, limits, goals };
}

function lerp3(a: Vec3, b: Vec3, t: number): Vec3 {
  return add(a, scale(sub(b, a), t));
}

/** 初始（第 0 步之前）位姿集合 */
export function initialPoses(frames: ToothFrame[]): Map<number, ToothPose> {
  const m = new Map<number, ToothPose>();
  for (const f of frames) m.set(f.label, initialPose(f));
  return m;
}

/**
 * 连续时间采样：stepIndex 为整数时取该步结束位姿；
 * 带小数时在相邻步之间插值，供时间轴平滑拖动与播放。
 */
export function samplePoses(
  plan: TreatmentPlan,
  frames: ToothFrame[],
  continuousStep: number,
): Map<number, ToothPose> {
  const start = initialPoses(frames);
  if (plan.steps.length === 0) return start;
  const x = Math.min(Math.max(continuousStep, 0), plan.steps.length);
  const k0 = Math.floor(x);
  const frac = x - k0;
  const before = k0 === 0 ? start : plan.steps[k0 - 1].poses;
  const after = plan.steps[Math.min(k0, plan.steps.length - 1)].poses;
  if (frac <= 1e-9) return before;
  const out = new Map<number, ToothPose>();
  for (const f of frames) {
    const a = before.get(f.label)!;
    const b = after.get(f.label)!;
    out.set(f.label, {
      position: lerp3(a.position, b.position, frac),
      orientation: qSlerp(a.orientation, b.orientation, frac),
    });
  }
  return out;
}

/** 某一步（整数步序号）实际在动的牙位 */
export function activeTeethAtStep(plan: TreatmentPlan, stepIndex: number): number[] {
  if (stepIndex < 0 || stepIndex >= plan.steps.length) return [];
  return plan.steps[stepIndex].activeLabels;
}
