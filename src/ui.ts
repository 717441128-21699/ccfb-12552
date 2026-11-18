// DOM 控制面板绑定
export interface UIHandlers {
  onWind(u0: number): void;
  onViscosity(nu: number): void;
  onSmoke(on: boolean): void;
  onSlice(on: boolean): void;
  onSlicePos(frac: number): void;
  onPause(paused: boolean): void;
  onReset(): void;
  onCarParam(): void; // 参数化车型任一参数变化(值由 getCarParams 取)
  onSaveA(): void;
  onSaveB(): void;
  onLoadFile(f: File): void;
  onFlip(): void;
  onRestoreDefault(): void;
}

export interface CarParamValues {
  mirrorPos: number;
  spoilerAngle: number;
  spoilerOn: boolean;
  rideHeight: number;
}

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;

export class UI {
  private paused = false;

  constructor(private h: UIHandlers) {
    const bindSlider = (id: string, vid: string, fmt: (v: number) => string, fn: (v: number) => void) => {
      const el = $<HTMLInputElement>(id);
      const out = $(vid);
      const update = () => {
        const v = parseFloat(el.value);
        out.textContent = fmt(v);
        fn(v);
      };
      el.addEventListener('input', update);
      update();
    };
    bindSlider('s-wind', 'v-wind', (v) => v.toFixed(3), (v) => h.onWind(v));
    bindSlider('s-nu', 'v-nu', (v) => v.toFixed(4), (v) => h.onViscosity(v));
    bindSlider('s-slice', 'v-slice', (v) => (v * 100).toFixed(0) + '%', (v) => h.onSlicePos(v));
    bindSlider('s-mirror', 'v-mirror', (v) => (v * 100).toFixed(0) + '%', () => h.onCarParam());
    bindSlider('s-spoiler', 'v-spoiler', (v) => v.toFixed(0) + '°', () => h.onCarParam());
    bindSlider('s-ride', 'v-ride', (v) => (v >= 0 ? '+' : '') + (v * 100).toFixed(0) + 'cm', () => h.onCarParam());

    $<HTMLInputElement>('c-smoke').addEventListener('change', (e) =>
      h.onSmoke((e.target as HTMLInputElement).checked),
    );
    $<HTMLInputElement>('c-slice').addEventListener('change', (e) =>
      h.onSlice((e.target as HTMLInputElement).checked),
    );
    $<HTMLInputElement>('c-spoiler').addEventListener('change', () => h.onCarParam());

    $('btn-pause').addEventListener('click', () => {
      this.paused = !this.paused;
      $('btn-pause').textContent = this.paused ? '继续' : '暂停';
      h.onPause(this.paused);
    });
    $('btn-reset').addEventListener('click', () => h.onReset());
    $('btn-a').addEventListener('click', () => h.onSaveA());
    $('btn-b').addEventListener('click', () => h.onSaveB());
    $('btn-flip').addEventListener('click', () => h.onFlip());
    $('btn-default').addEventListener('click', () => h.onRestoreDefault());
    $('btn-load').addEventListener('click', () => $<HTMLInputElement>('file').click());
    $<HTMLInputElement>('file').addEventListener('change', (e) => {
      const f = (e.target as HTMLInputElement).files?.[0];
      if (f) h.onLoadFile(f);
      (e.target as HTMLInputElement).value = '';
    });
  }

  getCarParams(): CarParamValues {
    return {
      mirrorPos: parseFloat($<HTMLInputElement>('s-mirror').value),
      spoilerAngle: parseFloat($<HTMLInputElement>('s-spoiler').value),
      spoilerOn: $<HTMLInputElement>('c-spoiler').checked,
      rideHeight: parseFloat($<HTMLInputElement>('s-ride').value),
    };
  }

  // 导入外部模型后禁用参数化车型的滑杆
  setProceduralEnabled(on: boolean) {
    for (const el of document.querySelectorAll('.proc')) el.classList.toggle('disabled', !on);
  }

  setStats(s: { grid: string; stepsPerSec: string; re: string; area: string }) {
    $('st-grid').textContent = s.grid;
    $('st-steps').textContent = s.stepsPerSec;
    $('st-re').textContent = s.re;
    $('st-area').textContent = s.area;
  }

  setCd(cd: number | null, cl: number | null) {
    $('st-cd').textContent = cd === null ? '—' : cd.toFixed(3);
    $('st-cl').textContent = cl === null ? '—' : cl.toFixed(3);
  }

  setCompare(a: number | null, b: number | null) {
    $('cd-a').textContent = a === null ? '—' : a.toFixed(3);
    $('cd-b').textContent = b === null ? '—' : b.toFixed(3);
    const d = $('cd-delta');
    if (a !== null && b !== null) {
      const delta = b - a;
      d.textContent = `${delta >= 0 ? '+' : ''}${delta.toFixed(3)} (${((delta / a) * 100).toFixed(1)}%)`;
      d.className = delta >= 0 ? 'pos' : 'neg';
      d.id = 'cd-delta';
    } else {
      d.textContent = '—';
      d.className = '';
    }
  }

  toast(msg: string, ms = 2600) {
    const t = $('toast');
    t.textContent = msg;
    t.style.opacity = '1';
    setTimeout(() => (t.style.opacity = '0'), ms);
  }
}

export function fatal(msg: string) {
  const el = $('fatal');
  el.textContent = msg;
  el.style.display = 'flex';
}
