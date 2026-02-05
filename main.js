// WebGPU 3D Gaussian Splatting Viewer
const canvas = document.getElementById('canvas');
const status = document.getElementById('status');
const fpsEl = document.getElementById('fps');
const loadBtn = document.getElementById('loadBtn');
const fileInput = document.getElementById('fileInput');

let device, context, format;
let camera = { x: 0, y: 0, z: 5, rotX: 0.5, rotY: 0 };
let isDragging = false;
let lastMouse = { x: 0, y: 0 };
let animating = false;

let vertexBuffer = null;
let uniformBuffer = null;
let bindGroup = null;
let pipeline = null;
let vertexCount = 0;
let depthTexture = null;
let frameCount = 0;
let fpsUpdateTime = 0;
let pointSizeMultiplier = 5;

// Init WebGPU
async function initWebGPU() {
  if (!navigator.gpu) {
    status.textContent = 'WebGPU not supported';
    status.style.color = '#f00';
    return false;
  }
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  device = await adapter.requestDevice();
  context = canvas.getContext('webgpu');
  format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode: 'opaque' });
  resize();
  return true;
}

function resize() {
  canvas.width = window.innerWidth * devicePixelRatio;
  canvas.height = window.innerHeight * devicePixelRatio;
  if (device) {
    depthTexture = device.createTexture({
      size: [canvas.width, canvas.height],
      format: 'depth24plus',
      usage: GPUTextureUsage.RENDER_ATTACHMENT
    });
  }
}

const shaderCode = `
struct Uniforms { mvp: mat4x4f, pointSize: f32, aspect: f32 }
@group(0) @binding(0) var<uniform> u: Uniforms;
struct VSOut { @builtin(position) pos: vec4f, @location(0) col: vec4f, @location(1) uv: vec2f }

@vertex fn vs(@location(0) p: vec3f, @location(1) c: vec4f, @builtin(vertex_index) vid: u32) -> VSOut {
  var o: VSOut;
  let basePos = u.mvp * vec4f(p, 1.0);
  // 6 vertices per quad (2 triangles)
  let quadIdx = vid % 6u;
  let size = u.pointSize * 0.001;
  var offset = vec2f(0.0);
  var uv = vec2f(0.0);
  // Triangle 1: 0,1,2  Triangle 2: 0,2,3
  if (quadIdx == 0u) { offset = vec2f(-1.0, -1.0); uv = vec2f(-1.0, -1.0); }
  else if (quadIdx == 1u) { offset = vec2f(1.0, -1.0); uv = vec2f(1.0, -1.0); }
  else if (quadIdx == 2u) { offset = vec2f(1.0, 1.0); uv = vec2f(1.0, 1.0); }
  else if (quadIdx == 3u) { offset = vec2f(-1.0, -1.0); uv = vec2f(-1.0, -1.0); }
  else if (quadIdx == 4u) { offset = vec2f(1.0, 1.0); uv = vec2f(1.0, 1.0); }
  else { offset = vec2f(-1.0, 1.0); uv = vec2f(-1.0, 1.0); }
  o.pos = basePos + vec4f(offset.x * size / u.aspect, offset.y * size, 0.0, 0.0);
  o.col = c;
  o.uv = uv;
  return o;
}

@fragment fn fs(in: VSOut) -> @location(0) vec4f {
  let d = length(in.uv);
  if (d > 1.0) { discard; }
  // Gaussian falloff for soft splat
  let alpha = exp(-d * d * 2.0);
  return vec4f(in.col.rgb, in.col.a * alpha);
}
`;

function createPipeline() {
  const shader = device.createShaderModule({ code: shaderCode });
  uniformBuffer = device.createBuffer({ size: 96, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const layout = device.createBindGroupLayout({
    entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX, buffer: {} }]
  });
  bindGroup = device.createBindGroup({ layout, entries: [{ binding: 0, resource: { buffer: uniformBuffer } }] });
  pipeline = device.createRenderPipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
    vertex: {
      module: shader, entryPoint: 'vs',
      buffers: [{ arrayStride: 28, attributes: [
        { shaderLocation: 0, offset: 0, format: 'float32x3' },
        { shaderLocation: 1, offset: 12, format: 'float32x4' }
      ]}]
    },
    fragment: { module: shader, entryPoint: 'fs', targets: [{ 
      format,
      blend: {
        color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
        alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' }
      }
    }] },
    primitive: { topology: 'triangle-list' },
    depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' }
  });
}

function perspective(fov, aspect, near, far) {
  const f = 1 / Math.tan(fov / 2);
  return new Float32Array([f/aspect,0,0,0, 0,f,0,0, 0,0,(far+near)/(near-far),-1, 0,0,2*far*near/(near-far),0]);
}

function lookAt() {
  const cx = Math.cos(camera.rotX), sx = Math.sin(camera.rotX);
  const cy = Math.cos(camera.rotY), sy = Math.sin(camera.rotY);
  return new Float32Array([cy,sx*sy,-cx*sy,0, 0,cx,sx,0, sy,-sx*cy,cx*cy,0, 0,0,-camera.z,1]);
}

function mulMat4(a, b) {
  const o = new Float32Array(16);
  for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++)
    o[i*4+j] = a[i*4]*b[j] + a[i*4+1]*b[4+j] + a[i*4+2]*b[8+j] + a[i*4+3]*b[12+j];
  return o;
}

// 生成兔子点云
function generateScene() {
  const pts = [];
  const add = (cx,cy,cz,rx,ry,rz,n,col) => {
    for (let i = 0; i < n; i++) {
      const u = Math.random(), v = Math.random();
      const theta = 2 * Math.PI * u, phi = Math.acos(2 * v - 1);
      const vary = 0.9 + Math.random() * 0.2;
      pts.push({
        x: cx + rx * Math.sin(phi) * Math.cos(theta),
        y: cy + ry * Math.cos(phi),
        z: cz + rz * Math.sin(phi) * Math.sin(theta),
        r: Math.min(1, col[0] * vary), 
        g: Math.min(1, col[1] * vary), 
        b: Math.min(1, col[2] * vary)
      });
    }
  };
  
  // 兔子 - 紧凑的坐标
  add(0, 0, 0, 0.5, 0.4, 0.6, 10000, [0.95, 0.9, 0.85]);      // 身体
  add(0, 0.45, 0.4, 0.3, 0.3, 0.32, 6000, [0.97, 0.92, 0.87]); // 头
  add(-0.12, 0.85, 0.35, 0.07, 0.28, 0.05, 2500, [1, 0.85, 0.9]); // 左耳
  add(0.12, 0.85, 0.35, 0.07, 0.28, 0.05, 2500, [1, 0.85, 0.9]);  // 右耳
  add(0, 0.12, -0.6, 0.14, 0.14, 0.12, 2000, [1, 0.98, 0.95]);    // 尾巴
  add(-0.18, -0.4, 0.25, 0.1, 0.18, 0.1, 1500, [0.92, 0.87, 0.82]); // 左前腿
  add(0.18, -0.4, 0.25, 0.1, 0.18, 0.1, 1500, [0.92, 0.87, 0.82]);  // 右前腿
  add(-0.22, -0.35, -0.3, 0.14, 0.2, 0.18, 2000, [0.92, 0.87, 0.82]); // 左后腿
  add(0.22, -0.35, -0.3, 0.14, 0.2, 0.18, 2000, [0.92, 0.87, 0.82]);  // 右后腿
  // 眼睛
  add(-0.1, 0.5, 0.65, 0.04, 0.04, 0.02, 500, [0.1, 0.1, 0.1]);
  add(0.1, 0.5, 0.65, 0.04, 0.04, 0.02, 500, [0.1, 0.1, 0.1]);
  // 鼻子
  add(0, 0.38, 0.7, 0.03, 0.02, 0.02, 300, [1, 0.6, 0.7]);
  
  return pts;
}

function generateDefaultCloud() {
  const points = generateScene();
  const count = points.length * 6; // 6 vertices per quad
  const data = new Float32Array(count * 7);
  for (let i = 0; i < points.length; i++) {
    const p = points[i];
    for (let j = 0; j < 6; j++) {
      const idx = (i * 6 + j) * 7;
      data[idx] = p.x; data[idx+1] = p.y; data[idx+2] = p.z;
      data[idx+3] = p.r; data[idx+4] = p.g; data[idx+5] = p.b; data[idx+6] = 1.0;
    }
  }
  return { data, count };
}

function loadCloud(data, count) {
  vertexCount = count;
  vertexBuffer = device.createBuffer({ size: data.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(vertexBuffer, 0, data);
  status.textContent = `${(count/6).toLocaleString()} points`;
}

function render(time) {
  if (!pipeline || !vertexBuffer) { requestAnimationFrame(render); return; }
  if (animating && !isDragging) camera.rotY += 0.005;
  frameCount++;
  if (time - fpsUpdateTime >= 1000) { fpsEl.textContent = `${frameCount} FPS`; frameCount = 0; fpsUpdateTime = time; }
  
  const aspect = canvas.width / canvas.height;
  const mvp = mulMat4(perspective(Math.PI / 4, aspect, 0.1, 100), lookAt());
  const uniformData = new Float32Array(20);
  uniformData.set(mvp, 0); uniformData[16] = pointSizeMultiplier; uniformData[17] = aspect;
  device.queue.writeBuffer(uniformBuffer, 0, uniformData);
  
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginRenderPass({
    colorAttachments: [{ view: context.getCurrentTexture().createView(), clearValue: { r: 0.02, g: 0.02, b: 0.03, a: 1 }, loadOp: 'clear', storeOp: 'store' }],
    depthStencilAttachment: { view: depthTexture.createView(), depthClearValue: 1.0, depthLoadOp: 'clear', depthStoreOp: 'store' }
  });
  pass.setPipeline(pipeline); pass.setBindGroup(0, bindGroup); pass.setVertexBuffer(0, vertexBuffer);
  pass.draw(vertexCount); pass.end();
  device.queue.submit([encoder.finish()]);
  requestAnimationFrame(render);
}

canvas.addEventListener('mousedown', e => { isDragging = true; lastMouse = { x: e.clientX, y: e.clientY }; });
canvas.addEventListener('mousemove', e => {
  if (!isDragging) return;
  camera.rotY += (e.clientX - lastMouse.x) * 0.01;
  camera.rotX += (e.clientY - lastMouse.y) * 0.01;
  camera.rotX = Math.max(-1.5, Math.min(1.5, camera.rotX));
  lastMouse = { x: e.clientX, y: e.clientY };
});
canvas.addEventListener('mouseup', () => isDragging = false);
canvas.addEventListener('mouseleave', () => isDragging = false);
canvas.addEventListener('wheel', e => { camera.z += e.deltaY * 0.005; camera.z = Math.max(1, Math.min(10, camera.z)); });

loadBtn.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', async e => {
  const file = e.target.files[0]; if (!file) return;
  status.textContent = 'Loading...';
  const buffer = await file.arrayBuffer();
  await parsePLY(buffer);
});

async function parsePLY(buffer) {
  const text = new TextDecoder().decode(buffer);
  const lines = text.split('\n');
  let pointCount = 0, headerEnd = 0;
  for (let i = 0; i < lines.length; i++) {
    if (lines[i].startsWith('element vertex')) pointCount = parseInt(lines[i].split(' ')[2]);
    if (lines[i] === 'end_header') { headerEnd = i + 1; break; }
  }
  const vertCount = pointCount * 6;
  const data = new Float32Array(vertCount * 7);
  for (let i = 0; i < pointCount; i++) {
    const p = lines[headerEnd + i].trim().split(/\s+/);
    const x = parseFloat(p[0]), y = parseFloat(p[1]), z = parseFloat(p[2]);
    const r = (parseFloat(p[3]) || 128) / 255, g = (parseFloat(p[4]) || 128) / 255, b = (parseFloat(p[5]) || 128) / 255;
    for (let j = 0; j < 6; j++) {
      const idx = (i * 6 + j) * 7;
      data[idx] = x; data[idx+1] = y; data[idx+2] = z;
      data[idx+3] = r; data[idx+4] = g; data[idx+5] = b; data[idx+6] = 1.0;
    }
  }
  loadCloud(data, vertCount);
}

window.addEventListener('resize', resize);

async function init() {
  if (await initWebGPU()) {
    createPipeline();
    const { data, count } = generateDefaultCloud();
    loadCloud(data, count);
    requestAnimationFrame(render);
  }
}
init();

// 点大小滑块
const sizeSlider = document.getElementById('sizeSlider');
const sizeVal = document.getElementById('sizeVal');
sizeSlider.addEventListener('input', e => {
  pointSizeMultiplier = parseFloat(e.target.value);
  sizeVal.textContent = e.target.value;
});
// cache bust 1770271687
