// UI 接线：把面板控件绑定到 App 控制器
import { App } from './app';
import { makeGoal, type ToothGoal } from './core/planner';
import { toothName } from './core/mesh';

const $ = <T extends HTMLElement>(id: string): T => document.getElementById(id) as T;

const app = new App();
const canvas = $<HTMLCanvasElement>('canvas');
const timeline = $<HTMLInputElement>('timeline');
const stepLabel = $<HTMLSpanElement>('step-label');
const activeTeethEl = $<HTMLDivElement>('active-teeth');
const iprInfo = $<HTMLDivElement>('ipr-info');
const caseInfo = $<HTMLDivElement>('case-info');
const segInfo = $<HTMLDivElement>('seg-info');
const gpuInfo = $<HTMLDivElement>('gpu-info');
const toothSelect = $<HTMLSelectElement>('sel-tooth');

app.onInfo = (msg) => { caseInfo.textContent = msg; };
app.onStepChange = (step, total, active) => {
  timeline.max = String(total);
  timeline.value = String(step);
  stepLabel.textContent = total ? `第 ${Math.min(Math.floor(step) + 1, total)} / ${total} 步` : '未生成方案';
  activeTeethEl.textContent = active.length
    ? '本步在动：' + active.map((l) => toothName(l)).join('、')
    : '本步无移动牙';
};
app.onIpr = (report) => {
  if (report.pairs.length === 0) { iprInfo.textContent = '无相邻牙对'; return; }
  const rows = report.pairs
    .filter((p) => p.required > 1e-6)
    .map((p) => `${p.labelA}-${p.labelB}: 重叠${p.overlap.toFixed(2)} → 各磨${p.stripA.toFixed(2)}/${p.stripB.toFixed(2)}mm` +
      (p.shortfall > 0 ? ` ⚠缺口${p.shortfall.toFixed(2)}` : ''));
  iprInfo.innerHTML = `总去釉 <b>${report.totalStrip.toFixed(2)}mm</b>，单面最大 ${report.maxSingleSurface.toFixed(2)}mm<br>` +
    (rows.length ? rows.join('<br>') : '当前位姿无重叠，无需去釉');
};

function fillToothSelect(): void {
  if (!app.state) return;
  toothSelect.innerHTML = '';
  for (const f of app.state.frames) {
    const opt = document.createElement('option');
    opt.value = String(f.label);
    opt.textContent = toothName(f.label);
    toothSelect.appendChild(opt);
  }
  segInfo.textContent = `分割出 ${app.state.model.teeth.length} 颗牙` +
    (app.state.gumBinding ? `，牙龈 ${app.state.model.gum.vertexCount} 顶点` : '，无牙龈');
}

function syncGoalInputs(): void {
  if (!app.state) return;
  const label = Number(toothSelect.value);
  const g = app.state.goals.get(label) ?? makeGoal(label);
  $<HTMLInputElement>('in-tx').value = String(g.translation[0]);
  $<HTMLInputElement>('in-ty').value = String(g.translation[1]);
  $<HTMLInputElement>('in-tz').value = String(g.translation[2]);
  $<HTMLInputElement>('in-yaw').value = String(g.yawDeg);
  $<HTMLInputElement>('in-tilt').value = String(g.tiltDeg);
  $<HTMLInputElement>('in-torque').value = String(g.torqueDeg);
}

// ---- 导入 ----
$<HTMLInputElement>('file-stl').addEventListener('change', async (e) => {
  const file = (e.target as HTMLInputElement).files?.[0];
  if (!file) return;
  const labelFile = $<HTMLInputElement>('file-labels').files?.[0];
  const labelText = labelFile ? await labelFile.text() : undefined;
  await app.importSTL(await file.arrayBuffer(), labelText);
  fillToothSelect();
  syncGoalInputs();
});
$<HTMLButtonElement>('btn-demo').addEventListener('click', () => {
  app.loadDemo(); fillToothSelect(); syncGoalInputs();
});
$<HTMLButtonElement>('btn-testmodel').addEventListener('click', () => {
  app.loadTestModel(); fillToothSelect(); syncGoalInputs();
});

// ---- 规划 ----
toothSelect.addEventListener('change', syncGoalInputs);
$<HTMLButtonElement>('btn-apply').addEventListener('click', () => {
  const label = Number(toothSelect.value);
  const goal: ToothGoal = {
    label,
    translation: [
      Number($<HTMLInputElement>('in-tx').value),
      Number($<HTMLInputElement>('in-ty').value),
      Number($<HTMLInputElement>('in-tz').value),
    ],
    yawDeg: Number($<HTMLInputElement>('in-yaw').value),
    tiltDeg: Number($<HTMLInputElement>('in-tilt').value),
    torqueDeg: Number($<HTMLInputElement>('in-torque').value),
  };
  app.setGoal(label, goal);
  caseInfo.textContent = `已应用 ${toothName(label)} 的目标位`;
});
$<HTMLButtonElement>('btn-clear').addEventListener('click', () => {
  app.clearGoals();
  ['in-tx', 'in-ty', 'in-tz', 'in-yaw', 'in-tilt', 'in-torque'].forEach((id) => { $<HTMLInputElement>(id).value = '0'; });
  iprInfo.textContent = '';
});
$<HTMLButtonElement>('btn-plan').addEventListener('click', () => {
  if (!app.state) return;
  app.state.limits.maxTranslationPerStep = Number($<HTMLInputElement>('in-maxtrans').value);
  app.state.limits.maxRotationPerStepDeg = Number($<HTMLInputElement>('in-maxrot').value);
  app.state.limits.maxIPRPerSurface = Number($<HTMLInputElement>('in-maxipr').value);
  app.generatePlan();
});

// ---- 动画时间轴 ----
timeline.addEventListener('input', () => {
  app.pause();
  app.setStep(Number(timeline.value));
});
const playBtn = $<HTMLButtonElement>('btn-play');
playBtn.addEventListener('click', () => {
  if (app.isPlaying) { app.pause(); playBtn.textContent = '播放'; }
  else {
    if (app.step >= app.totalSteps) app.setStep(0);
    app.play(); playBtn.textContent = '暂停';
  }
});

// ---- 显示开关 ----
$<HTMLInputElement>('chk-gum').addEventListener('change', (e) => {
  app.showGum = (e.target as HTMLInputElement).checked;
  app.refreshView();
});
$<HTMLInputElement>('chk-sss').addEventListener('change', (e) => {
  if (app.renderer) { app.renderer.sssEnabled = (e.target as HTMLInputElement).checked; }
});

// ---- 启动 ----
async function main(): Promise<void> {
  const ok = await app.initRenderer(canvas);
  if (ok) {
    gpuInfo.textContent = 'WebGPU 已就绪，釉质 SSS 近似已启用';
    app.loadDemo();
    fillToothSelect();
    syncGoalInputs();
  } else {
    gpuInfo.textContent = 'WebGPU 不可用（核心几何/规划功能仍可通过测试运行）';
  }
}
void main();
