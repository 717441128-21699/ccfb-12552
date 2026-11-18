import { mat4LookAt, mat4Perspective, mat4Multiply, type Mat4, type Vec3 } from './math';

// 轨道相机:左键旋转,右键/中键平移,滚轮缩放
export class OrbitCamera {
  target: Vec3 = [100, 18, 48];
  dist = 240;
  theta = -0.7; // 水平角
  phi = 0.32; // 俯仰角
  private canvas: HTMLCanvasElement;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    let dragging = 0;
    let lx = 0, ly = 0;
    canvas.addEventListener('pointerdown', (e) => {
      dragging = e.button === 0 ? 1 : 2;
      lx = e.clientX; ly = e.clientY;
      canvas.setPointerCapture(e.pointerId);
    });
    canvas.addEventListener('pointermove', (e) => {
      if (!dragging) return;
      const dx = e.clientX - lx, dy = e.clientY - ly;
      lx = e.clientX; ly = e.clientY;
      if (dragging === 1) {
        this.theta -= dx * 0.006;
        this.phi = Math.min(1.45, Math.max(-0.2, this.phi + dy * 0.006));
      } else {
        const s = this.dist * 0.0012;
        const right: Vec3 = [Math.cos(this.theta), 0, -Math.sin(this.theta)];
        this.target[0] -= right[0] * dx * s;
        this.target[2] -= right[2] * dx * s;
        this.target[1] = Math.max(2, this.target[1] + dy * s);
      }
    });
    const stop = () => (dragging = 0);
    canvas.addEventListener('pointerup', stop);
    canvas.addEventListener('pointercancel', stop);
    canvas.addEventListener('wheel', (e) => {
      e.preventDefault();
      this.dist = Math.min(900, Math.max(40, this.dist * Math.exp(e.deltaY * 0.001)));
    }, { passive: false });
    canvas.addEventListener('contextmenu', (e) => e.preventDefault());
  }

  viewProj(aspect: number): Mat4 {
    const cp = Math.cos(this.phi), sp = Math.sin(this.phi);
    const eye: Vec3 = [
      this.target[0] + this.dist * cp * Math.sin(this.theta),
      this.target[1] + this.dist * sp,
      this.target[2] + this.dist * cp * Math.cos(this.theta),
    ];
    const view = mat4LookAt(eye, this.target, [0, 1, 0]);
    const proj = mat4Perspective((50 * Math.PI) / 180, aspect, 1, 3000);
    return mat4Multiply(proj, view);
  }
}
