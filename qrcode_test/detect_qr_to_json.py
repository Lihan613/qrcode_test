import cv2, json, os, numpy as np

IMG = "test_qr_graphene_WS2_hBN.png"
JSON_OUT = "result.json"
VIZ_OUT  = "result_viz.png"
DBG_BIN  = "debug_bin.png"

def try_detect(img):
    det = cv2.QRCodeDetector()
    ok, texts, points, _ = det.detectAndDecodeMulti(img)
    res = []
    if ok and points is not None and texts is not None:
        for t, pts in zip(texts, points):
            if not t:
                continue
            quad = [[int(x), int(y)] for x, y in pts.reshape(-1,2)]
            res.append({"text": t, "quad": quad})
    return res

def preprocess_variants(img):
    # Original image (BGR)
    yield "orig", img
    # Grayscale + adaptive threshold
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bin1 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY, 25, 5)
    yield "adapt", cv2.cvtColor(bin1, cv2.COLOR_GRAY2BGR)

def upscale(img, f):
    if f == 1:
        return img
    h, w = img.shape[:2]
    return cv2.resize(img, (w*f, h*f), interpolation=cv2.INTER_NEAREST)

def main():
    print("[INFO] reading:", os.path.abspath(IMG))
    img0 = cv2.imread(IMG)
    if img0 is None:
        print("[ERROR] failed to read image")
        return

    tried = 0
    for f in (1, 2, 3, 4):
        imf = upscale(img0, f)
        for tag, imv in preprocess_variants(imf):
            tried += 1
            res = try_detect(imv)
            print(f"[TRY] scale x{f}, {tag}: {len(res)} codes")
            if res:
                # If upscaled, scale coordinates back to original image size
                if f != 1:
                    for r in res:
                        r["quad"] = [[int(x/f), int(y/f)] for x, y in r["quad"]]
                payload = {"image": IMG, "qr_codes": res, "method": f"x{f}/{tag}"}
                with open(JSON_OUT, "w", encoding="utf-8") as fjs:
                    json.dump(payload, fjs, ensure_ascii=False, indent=2)
                print("[OK] JSON ->", os.path.abspath(JSON_OUT))

                # Draw visualization
                vis = img0.copy()
                for r in res:
                    p = [(int(x), int(y)) for x, y in r["quad"]]
                    for i in range(4):
                        cv2.line(vis, p[i], p[(i+1)%4], (0,255,0), 2)
                    cv2.putText(vis, r["text"], p[0], cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                cv2.imwrite(VIZ_OUT, vis)
                print("[OK] VIZ  ->", os.path.abspath(VIZ_OUT))
                return

    # If all failed: export last binarized image for debugging
    gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    bin1 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY, 25, 5)
    cv2.imwrite(DBG_BIN, bin1)
    payload = {"image": IMG, "qr_codes": [], "method": "none"}
    with open(JSON_OUT, "w", encoding="utf-8") as fjs:
        json.dump(payload, fjs, ensure_ascii=False, indent=2)
    print("[WARN] no QR detected; wrote", os.path.abspath(JSON_OUT))
    print("[DBG ] saved", os.path.abspath(DBG_BIN))

if __name__ == "__main__":
    main()
