import cv2
import numpy as np

def inspect_chart(image_bytes):
    arr=np.frombuffer(image_bytes,dtype=np.uint8)
    # Decode without assuming that the uploaded file is RGB/BGR: some PNGs
    # are grayscale (2-D) or contain an alpha channel (4-D).
    image=cv2.imdecode(arr,cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("Файл изображения не удалось прочитать. Поддерживаются PNG, JPG и WEBP.")
    if image.ndim == 2:
        image=cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 4:
        image=cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    elif image.ndim == 3 and image.shape[2] == 3:
        pass
    else:
        raise ValueError(f"Неподдерживаемый формат изображения: {image.shape}")
    h,w=image.shape[:2]
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    green=cv2.inRange(hsv,np.array([35,35,35]),np.array([95,255,255]))
    red1=cv2.inRange(hsv,np.array([0,45,45]),np.array([12,255,255]))
    red2=cv2.inRange(hsv,np.array([165,45,45]),np.array([180,255,255]))
    red=cv2.bitwise_or(red1,red2)
    colored=cv2.bitwise_or(green,red)
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    edges=cv2.Canny(gray,60,150)
    total=float(h*w)
    green_pct=float(cv2.countNonZero(green)/total*100)
    red_pct=float(cv2.countNonZero(red)/total*100)
    edge_pct=float(cv2.countNonZero(edges)/total*100)

    # Candidate candle bodies: connected colored components within chart-like size limits.
    kernel=np.ones((2,2),np.uint8)
    mask=cv2.morphologyEx(colored,cv2.MORPH_OPEN,kernel)
    count, labels, stats, centroids=cv2.connectedComponentsWithStats(mask,8)
    candidates=[]
    for i in range(1,count):
        x,y,bw,bh,area=stats[i]
        if 2 <= bw <= max(18,int(w*0.025)) and 3 <= bh <= int(h*0.35) and area >= 5:
            candidates.append({"x":int(x),"y":int(y),"width":int(bw),"height":int(bh),"pixels":int(area)})
    # Candidate horizontal support/resistance/indicator lines.
    lines=cv2.HoughLinesP(edges,1,np.pi/180,threshold=max(35,int(w*0.12)),
                          minLineLength=max(45,int(w*0.15)),maxLineGap=8)
    horizontal=[]
    if lines is not None:
        # OpenCV versions/builds may return HoughLinesP as (N, 1, 4)
        # or (N, 4). Normalize both forms before iterating.
        line_segments = np.asarray(lines).reshape(-1, 4)
        for item in line_segments:
            x1,y1,x2,y2=map(int,item)
            if abs(y2-y1)<=2 and abs(x2-x1)>=max(45,int(w*0.15)):
                horizontal.append({"y_pixel":int(round((y1+y2)/2)),"x_start":min(x1,x2),"x_end":max(x1,x2)})
    horizontal=sorted(horizontal,key=lambda x:x["y_pixel"])
    # Deduplicate near-identical y coordinates.
    dedup=[]
    for line in horizontal:
        if not dedup or abs(line["y_pixel"]-dedup[-1]["y_pixel"])>4:
            dedup.append(line)
    return {
        "image_width":w,"image_height":h,
        "green_pixel_share_pct":round(green_pct,2),
        "red_pixel_share_pct":round(red_pct,2),
        "edge_density_pct":round(edge_pct,2),
        "candidate_candle_components":len(candidates),
        "candidate_candle_boxes":candidates[:100],
        "candidate_horizontal_lines":dedup[:40],
        "visual_balance":"more green pixels" if green_pct>red_pct*1.15 else "more red pixels" if red_pct>green_pct*1.15 else "mixed / neutral",
        "limitations":["Компоненты являются кандидатами на свечные тела; линии индикаторов и элементы интерфейса могут ошибочно распознаваться как свечи.",
                       "Горизонтальные линии возвращаются в пикселях, без перевода в цену: для этого нужна калибровка оси цены или OCR.",
                       "Тикер и таймфрейм автоматически не подтверждаются.",
                       "Визуальный анализ не является самостоятельным торговым сигналом."]
    }
