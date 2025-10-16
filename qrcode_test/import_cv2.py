import cv2
import json

# 读取图片
img = cv2.imread("test_qr_rotated_60px.png")

# 初始化 QRCode 检测器
detector = cv2.QRCodeDetector()

# 检测并解码多个二维码
ok, texts, points, _ = detector.detectAndDecodeMulti(img)

results = []
if ok and points is not None:
    for txt, pts in zip(texts, points):
        if not txt:   # 有时能检测到位置但解码失败
            continue
        # 四个角点坐标
        poly = pts.reshape(-1, 2).tolist()
        results.append({"text": txt, "quad": poly})

# 打印结果
print(json.dumps(results, indent=2, ensure_ascii=False))

# 可视化结果
vis = img.copy()
if results:
    for r in results:
        p = [(int(x), int(y)) for x, y in r["quad"]]
        for i in range(4):
            cv2.line(vis, p[i], p[(i+1) % 4], (0,255,0), 2)
        cv2.putText(vis, r["text"], p[0], cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    cv2.imwrite("detected.png", vis)
    print("检测结果已保存到 detected.png")