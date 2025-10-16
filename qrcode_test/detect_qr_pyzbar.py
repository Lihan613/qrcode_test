from pyzbar.pyzbar import decode, ZBarSymbol
from PIL import Image
import json, os

IMG = "test_qr_graphene_WS2_hBN.png"
JSON_OUT = "result.json"

def main():
    img = Image.open(IMG)  # Read image with PIL
    codes = decode(img, symbols=[ZBarSymbol.QRCODE])
    results = []
    for c in codes:
        txt = c.data.decode("utf-8", errors="replace")
        poly = [[p.x, p.y] for p in (c.polygon or [])]
        rect = [c.rect.left, c.rect.top, c.rect.width, c.rect.height]
        results.append({"text": txt, "quad": poly, "rect": rect})

    with open(JSON_OUT, "w", encoding="utf-8") as f:
        json.dump({"image": IMG, "qr_codes": results, "method": "pyzbar"}, f, ensure_ascii=False, indent=2)
    print("[OK] JSON ->", os.path.abspath(JSON_OUT), "codes:", len(results))

if __name__ == "__main__":
    main()
