/** 列主序 mat4（与 WGSL mat4x4 内存布局一致）与简易轨道相机 */

export type Mat4 = Float32Array;
export type Vec3 = [number, number, number];

export function mat4Identity(): Mat4 {
  const m = new Float32Array(16);
  m[0] = m[5] = m[10] = m[15] = 1;
  return m;
}

export function mat4Multiply(a: Mat4, b: Mat4): Mat4 {
  const out = new Float32Array(16);
  for (let c = 0; c < 4; c++) {
    for (let r = 0; r < 4; r++) {
      let s = 0;
      for (let k = 0; k < 4; k++) s += a[k * 4 + r] * b[c * 4 + k];
      out[c * 4 + r] = s;
    }
  }
  return out;
}

export function mat4Perspective(fovy: number, aspect: number, near: number, far: number): Mat4 {
  const f = 1 / Math.tan(fovy / 2);
  const nf = 1 / (near - far);
  const m = new Float32Array(16);
  m[0] = f / aspect;
  m[5] = f;
  m[10] = far * nf;
  m[11] = -1;
  m[14] = far * near * nf;
  return m;
}

export function mat4LookAt(eye: Vec3, target: Vec3, up: Vec3): Mat4 {
  const sub = (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  const cross = (a: Vec3, b: Vec3): Vec3 => [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
  const norm = (a: Vec3): Vec3 => {
    const l = Math.hypot(a[0], a[1], a[2]) || 1;
    return [a[0] / l, a[1] / l, a[2] / l];
  };
  const z = norm(sub(eye, target));
  const x = norm(cross(up, z));
  const y = cross(z, x);
  const m = mat4Identity();
  m[0] = x[0]; m[4] = x[1]; m[8] = x[2];
  m[1] = y[0]; m[5] = y[1]; m[9] = y[2];
  m[2] = z[0]; m[6] = z[1]; m[10] = z[2];
  m[12] = -(x[0] * eye[0] + x[1] * eye[1] + x[2] * eye[2]);
  m[13] = -(y[0] * eye[0] + y[1] * eye[1] + y[2] * eye[2]);
  m[14] = -(z[0] * eye[0] + z[1] * eye[1] + z[2] * eye[2]);
  return m;
}

/** 均匀缩放 + 居中平移：lattice 坐标映射到以域中心为原点的世界坐标 */
export function mat4WorldFromLattice(nx: number, ny: number, nz: number, worldScale: number): Mat4 {
  const m = mat4Identity();
  m[0] = worldScale;
  m[5] = worldScale;
  m[10] = worldScale;
  m[12] = -0.5 * nx * worldScale;
  m[13] = -0.5 * ny * worldScale;
  m[14] = -0.5 * nz * worldScale;
  return m;
}

export class OrbitCamera {
  azimuth = -2.35; // 绕 y 角
  elevation = 0.32;
  distance = 26;
  target: Vec3 = [0, 1.2, 0];

  attach(canvas: HTMLCanvasElement): void {
    let dragging = false;
    let lastX = 0;
    let lastY = 0;
    canvas.addEventListener('pointerdown', (e) => {
      dragging = true;
      lastX = e.clientX;
      lastY = e.clientY;
      canvas.setPointerCapture(e.pointerId);
    });
    canvas.addEventListener('pointermove', (e) => {
      if (!dragging) return;
      this.azimuth += (e.clientX - lastX) * 0.008;
      this.elevation = Math.max(-1.3, Math.min(1.45, this.elevation + (e.clientY - lastY) * 0.008));
      lastX = e.clientX;
      lastY = e.clientY;
    });
    canvas.addEventListener('pointerup', () => { dragging = false; });
    canvas.addEventListener('wheel', (e) => {
      e.preventDefault();
      this.distance = Math.max(6, Math.min(80, this.distance * Math.exp(e.deltaY * 0.001)));
    }, { passive: false });
  }

  eye(): Vec3 {
    const ce = Math.cos(this.elevation);
    return [
      this.target[0] + this.distance * ce * Math.cos(this.azimuth),
      this.target[1] + this.distance * Math.sin(this.elevation),
      this.target[2] + this.distance * ce * Math.sin(this.azimuth),
    ];
  }

  viewProj(aspect: number): Mat4 {
    const proj = mat4Perspective((50 * Math.PI) / 180, aspect, 0.1, 500);
    const view = mat4LookAt(this.eye(), this.target, [0, 1, 0]);
    return mat4Multiply(proj, view);
  }

  /** 世界空间相机右向量（view 矩阵第一行），用于 ribbon 烟丝展开 */
  right(): Vec3 {
    const view = mat4LookAt(this.eye(), this.target, [0, 1, 0]);
    // lookAt 列主序：右向量在 m[0],m[4],m[8]
    const v = [view[0], view[4], view[8]];
    const l = Math.hypot(v[0], v[1], v[2]) || 1;
    return [v[0] / l, v[1] / l, v[2] / l];
  }

  preset(name: 'side' | 'top' | '3d'): void {
    if (name === 'side') {
      this.azimuth = -Math.PI / 2;
      this.elevation = 0.05;
    } else if (name === 'top') {
      this.azimuth = 0;
      this.elevation = 1.45;
    } else {
      this.azimuth = -2.35;
      this.elevation = 0.32;
    }
  }
}
