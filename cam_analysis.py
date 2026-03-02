import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# STEP 1: LOAD IMAGE
img = cv2.imread("camring.jpeg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# blur to remove noise
blur = cv2.GaussianBlur(gray, (5,5), 0)

# STEP 2: EDGE DETECTION
edges = cv2.Canny(blur, 50, 150)

# STEP 3: FIND CENTER USING HOUGH CIRCLE
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 200,
                           param1=100, param2=30,
                           minRadius=50, maxRadius=500)

if circles is not None:
    circles = np.uint16(np.around(circles))
    cx, cy, r = circles[0][0]
else:
    print("Circle not detected")
    exit()

print("Center:", cx, cy)

# STEP 4: GET EDGE POINTS
edge_points = np.column_stack(np.where(edges > 0))

angle_data = {i: [] for i in range(360)}

for y, x in edge_points:
    dx = x - cx
    dy = y - cy

    radius = np.sqrt(dx**2 + dy**2)
    angle = int(np.degrees(np.arctan2(dy, dx)) % 360)

    angle_data[angle].append(radius)

# STEP 5: CALCULATE DISTANCE
results = []

for angle in range(360):
    if len(angle_data[angle]) > 0:
        inner_r = min(angle_data[angle])
        outer_r = max(angle_data[angle])
        distance = outer_r - inner_r
        results.append([angle, inner_r, outer_r, distance])

df = pd.DataFrame(results, columns=["Angle", "Inner_R", "Outer_R", "Distance"])
print(df.head())
# -------- LENGTH VALUES (IMPORTANT) --------
avg_length = df["Distance"].mean()
min_length = df["Distance"].min()
max_length = df["Distance"].max()

print("\n===== CAM RING LENGTH ANALYSIS =====")
print(f"Average Thickness : {avg_length:.2f} pixels")
print(f"Minimum Thickness : {min_length:.2f} pixels")
print(f"Maximum Thickness : {max_length:.2f} pixels")


# STEP 6: GRAPH
plt.plot(df["Angle"], df["Distance"])
plt.title("Distance vs Angle")
plt.xlabel("Angle")
plt.ylabel("Distance (pixels)")
plt.show()

# STEP 7: SHOW OUTPUT
output = img.copy()
cv2.circle(output, (cx, cy), 5, (0,0,255), -1)

cv2.imshow("Edges", edges)
cv2.imshow("Result", output)
cv2.waitKey(0)
cv2.destroyAllWindows()

# SAVE DATA
df.to_csv("cam_ring_output.csv", index=False)
print("CSV saved successfully")
summary = pd.DataFrame({
    "Average": [avg_length],
    "Minimum": [min_length],
    "Maximum": [max_length]
})

summary.to_csv("cam_ring_summary.csv", index=False)
