# 자동차가 이동하며 여러 방향으로 레이저를 쏘고 건물 표면 좌표를 수집하여
# 3D 점군(Point Cloud) 생성
# 라이다는 물체를 면으로 보는 것이 아니라, 많은 점좌표(x, y)를 모아 환경을 재구성한다.
# 즉, [x, y, z] 이런 점으로 구성할 수 있다.

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

building = [
    (-20, 10, 10, 20, 0, 20),
    (10, 20, 15, 25, 0, 25),
    (-15, -5, 35, 45, 0, 18),
    (5, 18, 50, 60, 0, 30)
]

car_positions = []

for y in np.linspace(0, 60, 25):
    car_positions.append(np.array([0, y, 2]))

def simulate_lidar(car_pos):
    points = []
    horizontal_angles = np.linspace(-90, 90, 120)
    vertical_angles = np.linspace(-15, 15, 8)
    max_distance = 80

    for v_deg in vertical_angles:
        v_rad = np.deg2rad(v_deg)
        for h_deg in horizontal_angles:
            h_rad = np.deg2rad(h_deg)

            dx = np.cos(v_rad) * np.cos(h_rad)
            dy = np.cos(v_rad) * np.sin(h_rad)
            dz = np.sin(v_rad)

            for dist in np.arange(0.5, max_distance, 0.5):
                px = car_pos[0] + dx * dist
                py = car_pos[1] + dy * dist
                pz = car_pos[2] + dz * dist

                hit = False
                for (xmin, xmax, ymin, ymax, zmin, zmax) in building:
                    if xmin <= px <= xmax and ymin <= py <= ymax and zmin <= pz <= zmax:
                        points.append([px, py, pz])
                        hit = True
                        break
                if hit:
                    break

    return points   # ← 이게 핵심

all_points = []

for pos in car_positions:
    scan_points = simulate_lidar(pos)
    all_points.extend(scan_points)

all_points = np.array(all_points)
print(f'총 점군 수: {len(all_points)}')

fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(
    all_points[:, 0],
    all_points[:, 1],
    all_points[:, 2],
    s=1,
    c=all_points[:, 2],
    cmap='jet'
)

car_positions_np = np.array(car_positions)
ax.plot(
    car_positions_np[:, 0],
    car_positions_np[:, 1],
    car_positions_np[:, 2],
    color='black', linewidth=3, label='Car path'
)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Simple LiDAR Point Cloud')
ax.set_xlim(-40, 40)
ax.set_ylim(0, 80)
ax.set_zlim(0, 30)

plt.legend()
plt.show()