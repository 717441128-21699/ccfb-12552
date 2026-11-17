// 应用控制器：导入 -> 焊接 -> 分割 -> 牙坐标系 -> 规划 -> IPR -> 牙龈变形 -> 渲染/动画
import { parseSTL, parseFaceLabels, weldMesh, computeSmoothNormals } from './core/stl';
import { segmentModel } from './core/segmentation';
import { buildFrames, type ToothFrame } from './core/tooth-frame';
import {
  buildPlan, initialPoses, samplePoses,
  type ToothGoal, type TreatmentPlan,
} from './core/planner';
import { computeIPR, type IPRReport, type ToothPose } from './core/ipr';
import { buildGumBinding, deformGum, type GumBinding } from './core/gum-deform';
import { buildDemoArch, buildTwoToothTestModel } from './core/synthetic';
import type { Mesh, SegmentedModel, ToothPart } from './core/mesh';
import { WebGPURenderer, type RenderObject, poseToWorld } from './core/renderer';

export interface AppState {
  fullMesh: Mesh;
  model: SegmentedModel;
  frames: ToothFrame[];
  goals: Map<number, ToothGoal>;
  plan: TreatmentPlan | null;
  ipr: IPRReport | null;
  gumBinding: GumBinding | null;
  limits: { maxTranslationPerStep: number; maxRotationPerStepDeg: number; maxIPRPerSurface: number };
}

export class App {
  state: AppState | null = null;
  renderer: WebGPURenderer | null = null;
  showGum = true;
  private currentStep = 0;
  private playing = false;
  private lastFrameTime = 0;
  onStepChange?: (step: number, total: number, active: number[]) => void;
  onIpr?: (report: IPRReport) => void;
  onInfo?: (msg: string) => void;

  async initRenderer(canvas: HTMLCanvasElement): Promise<boolean> {
    if (!(await WebGPURenderer.isSupported())) {
      this.onInfo?.('当前浏览器不支持 WebGPU，请用 Chrome/Edge 113 以上版本');
      return false;
    }
    this.renderer = new WebGPURenderer(canvas);
    await this.renderer.init();
    this.attachOrbit(canvas);
    this.loop();
    return true;
  }

  private attachOrbit(canvas: HTMLCanvasElement): void {
    let dragging = false;
    let lastX = 0, lastY = 0;
    canvas.addEventListener('pointerdown', (e) => {
      dragging = true; lastX = e.clientX; lastY = e.clientY;
      canvas.setPointerCapture(e.pointerId);
    });
    canvas.addEventListener('pointermove', (e) => {
      if (!dragging || !this.renderer) return;
      const dx = e.clientX - lastX;
      const dy = e.clientY - lastY;
      this.renderer.orbit.yaw += dx * 0.01;
      this.renderer.orbit.pitch = Math.max(-1.4, Math.min(1.4, this.renderer.orbit.pitch + dy * 0.01));
      lastX = e.clientX; lastY = e.clientY;
    });
    canvas.addEventListener('pointerup', () => { dragging = false; });
    canvas.addEventListener('wheel', (e) => {
      if (!this.renderer) return;
      e.preventDefault();
      this.renderer.orbit.distance = Math.max(20, Math.min(250, this.renderer.orbit.distance * (1 + Math.sign(e.deltaY) * 0.08)));
    }, { passive: false });
  }

  loadFromRaw(raw: import('./core/mesh').RawMesh): void {
    const welded = weldMesh(raw);
    this.loadMesh(welded);
  }

  loadMesh(fullMesh: Mesh): void {
    const model = segmentModel(fullMesh);
    const frames = buildFrames(model.teeth);
    const gumBinding = model.gum.vertexCount > 0 ? buildGumBinding(model) : null;
    this.state = {
      fullMesh, model, frames,
      goals: new Map(),
      plan: null,
      ipr: null,
      gumBinding,
      limits: { maxTranslationPerStep: 0.3, maxRotationPerStepDeg: 2, maxIPRPerSurface: 0.5 },
    };
    this.currentStep = 0;
    this.refreshObjects(0);
    this.onInfo?.(`已载入：${model.teeth.length} 颗牙，${fullMesh.vertexCount} 顶点，${fullMesh.faceCount} 三角面`);
  }

  async importSTL(stlBuffer: ArrayBuffer, labelText?: string): Promise<void> {
    const labels = labelText ? parseFaceLabels(labelText) : undefined;
    const raw = parseSTL(stlBuffer, labels);
    if (!labels) {
      // 无标签文件：全部归为牙龈会无法分割，提示需要旁车标签
      this.onInfo?.('未提供标签文件：所有面将归为牙龈。请同时选择 .lab/.json 面标签文件。');
    }
    this.loadFromRaw(raw);
  }

  loadDemo(): void {
    const built = buildDemoArch();
    this.loadMesh(weldMesh(built.raw));
  }

  loadTestModel(): void {
    const built = buildTwoToothTestModel(4, 4, 8, -0.6);
    this.loadMesh(weldMesh(built.raw));
  }

  setGoal(label: number, goal: ToothGoal): void {
    if (!this.state) return;
    this.state.goals.set(label, goal);
  }

  clearGoals(): void {
    if (!this.state) return;
    this.state.goals.clear();
    this.state.plan = null;
    this.currentStep = 0;
    this.refreshObjects(0);
  }

  /** 生成分步方案并计算 IPR */
  generatePlan(): void {
    if (!this.state) return;
    const s = this.state;
    s.plan = buildPlan(s.frames, s.goals, {
      maxTranslationPerStep: s.limits.maxTranslationPerStep,
      maxRotationPerStepDeg: s.limits.maxRotationPerStepDeg,
    });
    // IPR 基于最终目标位姿计算
    const finalPoses = s.plan.steps.length
      ? s.plan.steps[s.plan.steps.length - 1].poses
      : initialPoses(s.frames);
    s.ipr = computeIPR(s.model.teeth, s.frames, finalPoses, {
      maxPerSurface: s.limits.maxIPRPerSurface,
    });
    this.currentStep = 0;
    this.onIpr?.(s.ipr);
    this.onStepChange?.(0, s.plan.steps.length, []);
    this.refreshObjects(0);
    this.onInfo?.(`生成 ${s.plan.steps.length} 步；IPR 总去釉 ${s.ipr.totalStrip.toFixed(2)}mm` +
      (s.ipr.capped ? '（有邻面达到 0.5mm 上限，存在缺口）' : ''));
  }

  get totalSteps(): number {
    return this.state?.plan?.steps.length ?? 0;
  }

  /** 拖动时间轴：continuousStep 可为小数（步间平滑插值） */
  setStep(continuousStep: number): void {
    this.currentStep = continuousStep;
    this.refreshObjects(continuousStep);
    const plan = this.state?.plan;
    if (plan && this.onStepChange) {
      const integerStep = Math.min(Math.floor(continuousStep), plan.steps.length - 1);
      const active = continuousStep <= 0 ? [] : plan.steps[integerStep]?.activeLabels ?? [];
      this.onStepChange(continuousStep, plan.steps.length, active);
    }
  }

  play(): void {
    if (this.playing || !this.state?.plan) return;
    this.playing = true;
    this.lastFrameTime = performance.now();
  }

  pause(): void { this.playing = false; }
  get isPlaying(): boolean { return this.playing; }
  get step(): number { return this.currentStep; }

  private loop = (): void => {
    requestAnimationFrame(this.loop);
    if (this.playing && this.state?.plan) {
      const now = performance.now();
      const dt = (now - this.lastFrameTime) / 1000;
      this.lastFrameTime = now;
      // 每秒走 1.5 步
      let next = this.currentStep + dt * 1.5;
      if (next >= this.state.plan.steps.length) {
        next = this.state.plan.steps.length;
        this.playing = false;
      }
      this.setStep(next);
    }
    this.renderer?.resize(this.canvasSize().w, this.canvasSize().h);
    this.renderer?.render();
  };

  private canvasSize(): { w: number; h: number } {
    const c = this.renderer?.canvas;
    return c ? { w: c.clientWidth, h: c.clientHeight } : { w: 1, h: 1 };
  }

  /** 根据当前连续步骤重建渲染对象（牙位姿 + 牙龈变形） */
  private refreshObjects(continuousStep: number): void {
    if (!this.state || !this.renderer) return;
    const s = this.state;
    const poses = s.plan
      ? samplePoses(s.plan, s.frames, continuousStep)
      : initialPoses(s.frames);

    const teeth: RenderObject[] = s.model.teeth.map((part) => {
      const frame = s.frames.find((f) => f.label === part.label)!;
      const pose = poses.get(part.label)!;
      const { positions, normals } = poseToWorld(part, frame.centroid, pose.position, pose.orientation);
      return { part, label: part.label, positions, normals, indices: part.indices };
    });

    let gum: RenderObject | null = null;
    if (this.showGum && s.gumBinding) {
      const deformed = deformGum(s.gumBinding, s.model.teeth, s.frames, poses);
      const dn = computeSmoothNormals(deformed, s.model.gum.indices);
      gum = {
        part: s.model.gum,
        label: 0,
        positions: deformed,
        normals: dn,
        indices: s.model.gum.indices,
        isGum: true,
      };
    }
    this.renderer.setObjects(teeth, gum);
  }

  refreshView(): void { this.refreshObjects(this.currentStep); }

  /** 导出当前全模型为二进制 STL（供外部工具核对） */
  exportSTL(): Blob {
    if (!this.state) throw new Error('无模型');
    const s = this.state;
    const poses = s.plan ? samplePoses(s.plan, s.frames, this.currentStep) : initialPoses(s.frames);
    const parts: { positions: Float32Array; indices: Uint32Array }[] = [];
    for (const part of s.model.teeth) {
      const frame = s.frames.find((f) => f.label === part.label)!;
      const pose = poses.get(part.label)!;
      parts.push({ ...poseToWorld(part, frame.centroid, pose.position, pose.orientation), indices: part.indices });
    }
    if (this.showGum && s.gumBinding) {
      const deformed = deformGum(s.gumBinding, s.model.teeth, s.frames, poses);
      parts.push({ positions: deformed, indices: s.model.gum.indices });
    }    return encodeBinarySTL(parts);
  }
}

/** 编码二进制 STL（面法线置零，由读取端重算） */
export function encodeBinarySTL(parts: { positions: Float32Array; indices: Uint32Array }[]): Blob {
  const triCount = parts.reduce((n, p) => n + p.indices.length / 3, 0);
  const buf = new ArrayBuffer(84 + triCount * 50);
  const dv = new DataView(buf);
  dv.setUint32(80, triCount, true);
  let off = 84;
  for (const part of parts) {
    for (let f = 0; f < part.indices.length / 3; f++) {
      off += 12; // 法线
      for (let k = 0; k < 3; k++) {
        const vi = part.indices[f * 3 + k];
        for (let c = 0; c < 3; c++) {
          dv.setFloat32(off, part.positions[vi * 3 + c], true);
          off += 4;
        }
      }
      dv.setUint16(off, 0, true);
      off += 2;
    }
  }
  return new Blob([buf], { type: 'model/stl' });
}

export type { ToothPart, ToothPose };
