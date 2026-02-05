// 内置模型：程序化生成的3D点云
export function generateBunny() {
  // 简化版兔子形状 - 用多个椭球组合
  const points = [];
  
  // 身体 (大椭球)
  addEllipsoid(points, 0, 0, 0, 0.4, 0.35, 0.5, 8000, [0.9, 0.85, 0.8]);
  
  // 头 (球)
  addEllipsoid(points, 0, 0.35, 0.35, 0.25, 0.25, 0.28, 5000, [0.92, 0.87, 0.82]);
  
  // 左耳
  addEllipsoid(points, -0.1, 0.7, 0.3, 0.06, 0.25, 0.04, 2000, [0.95, 0.8, 0.85]);
  
  // 右耳
  addEllipsoid(points, 0.1, 0.7, 0.3, 0.06, 0.25, 0.04, 2000, [0.95, 0.8, 0.85]);
  
  // 尾巴 (小球)
  addEllipsoid(points, 0, 0.1, -0.5, 0.12, 0.12, 0.1, 1500, [1, 0.95, 0.9]);
  
  // 左前腿
  addEllipsoid(points, -0.15, -0.35, 0.2, 0.08, 0.15, 0.08, 1000, [0.88, 0.83, 0.78]);
  
  // 右前腿
  addEllipsoid(points, 0.15, -0.35, 0.2, 0.08, 0.15, 0.08, 1000, [0.88, 0.83, 0.78]);
  
  // 左后腿
  addEllipsoid(points, -0.18, -0.3, -0.25, 0.12, 0.18, 0.15, 1500, [0.88, 0.83, 0.78]);
  
  // 右后腿
  addEllipsoid(points, 0.18, -0.3, -0.25, 0.12, 0.18, 0.15, 1500, [0.88, 0.83, 0.78]);

  return points;
}

function addEllipsoid(points, cx, cy, cz, rx, ry, rz, count, color) {
  for (let i = 0; i < count; i++) {
    const u = Math.random(), v = Math.random();
    const theta = 2 * Math.PI * u;
    const phi = Math.acos(2 * v - 1);
    
    const x = cx + rx * Math.sin(phi) * Math.cos(theta);
    const y = cy + ry * Math.cos(phi);
    const z = cz + rz * Math.sin(phi) * Math.sin(theta);
    
    // 添加一点颜色变化
    const variation = 0.95 + Math.random() * 0.1;
    points.push({
      x, y, z,
      r: color[0] * variation,
      g: color[1] * variation,
      b: color[2] * variation
    });
  }
}
