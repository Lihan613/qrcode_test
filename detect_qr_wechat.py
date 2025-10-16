import cv2, json, os

IMG = "test_qr_graphene_WS2_hBN.png"   # Image path
MODELS = {
    "det_proto": r".\models\detect.prototxt",
    "det_model": r".\models\detect.caffemodel",
    "sr_proto":  r".\models\sr.prototxt",
    "sr_model":  r".\models\sr.caffemodel",
}
JSON_OUT = "result.json"
VIZ_OUT  = "result_viz.png"

def main():
    # 1) Build WeChat QR detector
    det = cv2.wechat_qrcode_WeChatQRCode(
        MODELS["det_proto"], MODELS["det_model"],
        MODELS["sr_proto"],  MODELS["sr_model"]
    )

    img = cv2.imread(IMG)
    if img is None:
        print("[ERROR] read image fail:", os.path.abspath(IMG))
        return

    # 2) Directly detect and decode (multiple codes)
    texts, points = det.detectAndDecode(img)   # points: list of 4x2
    results = []
    if points is not None:
        for t, pts in zip(texts, points):
            if not t: 
                continue
            quad = [[int(x), int(y)] for x, y in pts.reshape(-1,2)]
            results.append({"text": t, "quad": quad})

    # 3) Write JSON file
    payload = {"image": IMG, "qr_codes": results, "method": "wechat_qrcode"}
    with open(JSON_OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print("[OK] JSON ->", os.path.abspath(JSON_OUT), "codes:", len(results))

    # 4) Visualization
    if results:
        vis = img.copy()
        for r in results:
            p = [(int(x), int(y)) for x, y in r["quad"]]
            for i in range(4):
                cv2.line(vis, p[i], p[(i+1)%4], (0,255,0), 2)
            cv2.putText(vis, r["text"], p[0], cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        cv2.imwrite(VIZ_OUT, vis)
        print("[OK] VIZ  ->", os.path.abspath(VIZ_OUT))
    else:
        print("[WARN] no QR decoded")

if __name__ == "__main__":
    main()
