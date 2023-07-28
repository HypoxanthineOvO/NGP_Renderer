import numpy as np

w, h = (800, 800)
focal_length = .5 * w / np.tan(.5 * 0.6911112070083618)
inv_focal_length = 1 / focal_length

i, j = np.meshgrid(
            np.linspace(0, w-1, w), 
            np.linspace(0, h-1, h), 
            indexing='xy'
        )
directions = np.stack([(i-0.5 * w)/ focal_length, -(j-0.5 * h)/focal_length, -np.ones_like(i)], -1)
print(directions[:3, :3])

directions_2 = np.zeros([w, h, 3], dtype = np.float32)
for i in range(w):
    for j in range(h):
        directions_2[j, i] = np.array([(i-0.5*w) * inv_focal_length, -(j-0.5*h) * inv_focal_length, -1])
print(directions_2[:3, :3])

delta = directions - directions_2
print(delta.sum())