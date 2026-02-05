// WebGPU 3D Gaussian Splatting Viewer
const canvas = document.getElementById('canvas');
const status = document.getElementById('status');
const fpsEl = document.getElementById('fps');
const loadBtn = document.getElementById('loadBtn');
const fileInput = document.getElementById('fileInput');

let device, context, format;
let camera = { x: 0, y: 0, z: 3, rotX: 0.3, rotY: 0.5 };
let isDragging = false;
let lastMouse = { x: 0, y: 0 };
let animating = true;

// Buffers
let vertexBuffer = null;
let uniformBuffer = null;
let bindGroup = null;
let pipeline = null;
let vertexCount = 0;
let depthTexture = null;

// 性能优化：使用 RAF 时间戳
let lastFrameTime = 0;
let frameCount = 0;
let fpsUpdateTime = 0;

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
  
  // 重建深度纹理
  if (device) {
    depthTexture = device.createTexture({
      size: [canvas.width, canvas.height],
      format: 'depth24plus',
      usage: GPUTextureUsage.RENDER_ATTACHMENT
    });
  }
}

// Shader - 使用小三角形代替点（WebGPU 不支持 point_size）
const shaderCode = `
struct Uniforms {
  mvp: mat4x4f,
  pointSize: f32,
  aspect: f32,
}
@group(0) @binding(0) var<uniform> u: Uniforms;

struct VSOut {
  @builtin(position) pos: vec4f,
  @location(0) col: vec4f,
}

@vertex fn vs(@location(0) p: vec3f, @location(1) c: vec4f, @builtin(vertex_index) vid: u32) -> VSOut {
  var o: VSOut;
  let basePos = u.mvp * vec4f(p, 1.0);
  
  // 每个点用3个顶点画一个小三角形
  let triIdx = vid % 3u;
  let size = u.pointSize * 0.008;
  var offset = vec2f(0.0, 0.0);
  if (triIdx == 0u) { offset = vec2f(0.0, size); }
  else if (triIdx == 1u) { offset = vec2f(-size * 0.866, -size * 0.5); }
  else { offset = vec2f(size * 0.866, -size * 0.5); }
  
  o.pos = basePos + vec4f(offset.x / u.aspect, offset.y, 0.0, 0.0);
  o.col = c;
  return o;
}

@fragment fn fs(in: VSOut) -> @location(0) vec4f {
  return in.col;
}
`;

// 创建渲染管线
function createPipeline() {
  const shader = device.createShaderModule({ code: shaderCode });
  
  uniformBuffer = device.createBuffer({
    size: 96,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  
  const layout = device.createBindGroupLayout({
    entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX, buffer: {} }]
  });
  
  bindGroup = device.createBindGroup({
    layout,
    entries: [{ binding: 0, resource: { buffer: uniformBuffer } }]
  });
  
  pipeline = device.createRenderPipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
    vertex: {
      module: shader,
      entryPoint: 'vs',
      buffers: [{
        arrayStride: 28,
        attributes: [
          { shaderLocation: 0, offset: 0, format: 'float32x3' },
          { shaderLocation: 1, offset: 12, format: 'float32x4' },
        ]
      }]
    },
    fragment: {
      module: shader,
      entryPoint: 'fs',
      targets: [{ format }]
    },
    primitive: { topology: 'triangle-list' },
    depthStencil: {
      format: 'depth24plus',
      depthWriteEnabled: true,
      depthCompare: 'less'
    }
  });
}

// 矩阵运算
function perspective(fov, aspect, near, far) {
  const f = 1 / Math.tan(fov / 2);
  return new Float32Array([
    f/aspect, 0, 0, 0,
    0, f, 0, 0,
    0, 0, (far+near)/(near-far), -1,
    0, 0, 2*far*near/(near-far), 0
  ]);
}

function lookAt() {
  const cx = Math.cos(camera.rotX), sx = Math.sin(camera.rotX);
  const cy = Math.cos(camera.rotY), sy = Math.sin(camera.rotY);
  return new Float32Array([
    cy, sx*sy, -cx*sy, 0,
    0, cx, sx, 0,
    sy, -sx*cy, cx*cy, 0,
    0, 0, -camera.z, 1
  ]);
}

function mulMat4(a, b) {
  const o = new Float32Array(16);
  for (let i = 0; i < 4; i++) {
    for (let j = 0; j < 4; j++) {
      o[i*4+j] = a[i*4]*b[j] + a[i*4+1]*b[4+j] + a[i*4+2]*b[8+j] + a[i*4+3]*b[12+j];
    }
  }
  return o;
}

// 生成默认点云 - 彩色球体（每个点3个顶点）
function generateDefaultCloud() {
  const pointCount = 50000;
  const count = pointCount * 3; // 每个点3个顶点
  const data = new Float32Array(count * 7);
  
  for (let i = 0; i < pointCount; i++) {
    // 球面均匀分布
    const u = Math.random(), v = Math.random();
    const theta = 2 * Math.PI * u;
    const phi = Math.acos(2 * v - 1);
    const r = 0.8 + Math.random() * 0.2;
    
    const x = r * Math.sin(phi) * Math.cos(theta);
    const y = r * Math.sin(phi) * Math.sin(theta);
    const z = r * Math.cos(phi);
    
    // HSL 颜色基于位置
    const h = (Math.atan2(y, x) + Math.PI) / (2 * Math.PI);
    const s = 0.8, l = 0.6;
    const c = (1 - Math.abs(2*l - 1)) * s;
    const xc = c * (1 - Math.abs((h*6) % 2 - 1));
    const m = l - c/2;
    let r1, g1, b1;
    const hi = Math.floor(h * 6);
    if (hi === 0) { r1=c; g1=xc; b1=0; }
    else if (hi === 1) { r1=xc; g1=c; b1=0; }
    else if (hi === 2) { r1=0; g1=c; b1=xc; }
    else if (hi === 3) { r1=0; g1=xc; b1=c; }
    else if (hi === 4) { r1=xc; g1=0; b1=c; }
    else { r1=c; g1=0; b1=xc; }
    
    // 每个点复制3次（3个顶点）
    for (let j = 0; j < 3; j++) {
      const idx = (i * 3 + j) * 7;
      data[idx] = x;
      data[idx+1] = y;
      data[idx+2] = z;
      data[idx+3] = r1 + m;
      data[idx+4] = g1 + m;
      data[idx+5] = b1 + m;
      data[idx+6] = 1.0;
    }
  }
  
  return { data, count };
}

// 加载点云数据到 GPU
function loadCloud(data, count) {
  vertexCount = count;
  vertexBuffer = device.createBuffer({
    size: data.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(vertexBuffer, 0, data);
  status.textContent = `${count.toLocaleString()} points`;
}

// 渲染循环
function render(time) {
  if (!pipeline || !vertexBuffer) {
    requestAnimationFrame(render);
    return;
  }
  
  // 自动旋转
  if (animating && !isDragging) {
    camera.rotY += 0.005;
  }
  
  // FPS 计算
  frameCount++;
  if (time - fpsUpdateTime >= 1000) {
    fpsEl.textContent = `${frameCount} FPS`;
    frameCount = 0;
    fpsUpdateTime = time;
  }
  
  // MVP 矩阵
  const aspect = canvas.width / canvas.height;
  const proj = perspective(Math.PI / 4, aspect, 0.1, 100);
  const view = lookAt();
  const mvp = mulMat4(proj, view);
  
  // 更新 uniform
  const uniformData = new Float32Array(20);
  uniformData.set(mvp, 0);
  uniformData[16] = 3.0; // pointSize
  uniformData[17] = aspect; // aspect ratio
  device.queue.writeBuffer(uniformBuffer, 0, uniformData);
  
  // 渲染
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginRenderPass({
    colorAttachments: [{
      view: context.getCurrentTexture().createView(),
      clearValue: { r: 0.02, g: 0.02, b: 0.03, a: 1 },
      loadOp: 'clear',
      storeOp: 'store'
    }],
    depthStencilAttachment: {
      view: depthTexture.createView(),
      depthClearValue: 1.0,
      depthLoadOp: 'clear',
      depthStoreOp: 'store'
    }
  });
  
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.setVertexBuffer(0, vertexBuffer);
  pass.draw(vertexCount);
  pass.end();
  
  device.queue.submit([encoder.finish()]);
  requestAnimationFrame(render);
}

// 鼠标控制
canvas.addEventListener('mousedown', e => {
  isDragging = true;
  lastMouse = { x: e.clientX, y: e.clientY };
});

canvas.addEventListener('mousemove', e => {
  if (!isDragging) return;
  camera.rotY += (e.clientX - lastMouse.x) * 0.01;
  camera.rotX += (e.clientY - lastMouse.y) * 0.01;
  camera.rotX = Math.max(-1.5, Math.min(1.5, camera.rotX));
  lastMouse = { x: e.clientX, y: e.clientY };
});

canvas.addEventListener('mouseup', () => isDragging = false);
canvas.addEventListener('mouseleave', () => isDragging = false);

canvas.addEventListener('wheel', e => {
  camera.z += e.deltaY * 0.005;
  camera.z = Math.max(1, Math.min(10, camera.z));
});

// PLY 文件加载
loadBtn.addEventListener('click', () => fileInput.click());

fileInput.addEventListener('change', async e => {
  const file = e.target.files[0];
  if (!file) return;
  status.textContent = 'Loading...';
  const buffer = await file.arrayBuffer();
  await parsePLY(buffer);
});

async function parsePLY(buffer) {
  const text = new TextDecoder().decode(buffer);
  const lines = text.split('\n');
  
  let pointCount = 0, headerEnd = 0;
  for (let i = 0; i < lines.length; i++) {
    if (lines[i].startsWith('element vertex')) {
      pointCount = parseInt(lines[i].split(' ')[2]);
    }
    if (lines[i] === 'end_header') {
      headerEnd = i + 1;
      break;
    }
  }
  
  // 每个点需要3个顶点来画三角形
  const vertCount = pointCount * 3;
  const data = new Float32Array(vertCount * 7);
  
  for (let i = 0; i < pointCount; i++) {
    const p = lines[headerEnd + i].trim().split(/\s+/);
    const x = parseFloat(p[0]);
    const y = parseFloat(p[1]);
    const z = parseFloat(p[2]);
    const r = (parseFloat(p[3]) || 128) / 255;
    const g = (parseFloat(p[4]) || 128) / 255;
    const b = (parseFloat(p[5]) || 128) / 255;
    
    // 复制3次（3个顶点组成三角形）
    for (let j = 0; j < 3; j++) {
      const idx = (i * 3 + j) * 7;
      data[idx] = x;
      data[idx+1] = y;
      data[idx+2] = z;
      data[idx+3] = r;
      data[idx+4] = g;
      data[idx+5] = b;
      data[idx+6] = 1.0;
    }
  }
  
  loadCloud(data, vertCount);
  status.textContent = `${pointCount.toLocaleString()} points loaded`;
}

window.addEventListener('resize', resize);

// 初始化
async function init() {
  if (await initWebGPU()) {
    createPipeline();
    const { data, count } = generateDefaultCloud();
    loadCloud(data, count);
    requestAnimationFrame(render);
  }
}

init();
