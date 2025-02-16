import datetime
import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Voleybol oyuncularının isimlerini ID'leri ile eşleştiren sözlük
# Bu sayede tespit edilen her oyuncunun ismi ekranda görüntülenebilecek
class_labels = {
    0: "hande-baladin",
    1: "cansu-ozbay",
    2: "melissa-vargas",
    3: "eda-erdem",
    4: "zehra-gunes",
    5: "ebrar-karakurt",
    6: "gizem-orge",
    7: "simge-akoz",
    8: "saliha-sahin",
    9: "ilkin-aydin",
    10: "daniele-santarelli",
    11: "ayca-aykac",
    12: "elif-sahin",
    13: "derya-cebecioglu",
    14: "asli-kalac",
    15: "kubra-akman",
}

# Temel ayarlar
# CONFIDENCE_THRESHOLD: Bir tespitin geçerli sayılması için gereken minimum güven değeri
# GREEN ve WHITE: Ekranda gösterilecek dikdörtgen ve yazılar için renk değerleri
CONFIDENCE_THRESHOLD = 0.6
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)

# Video dosyasını aç
video_cap = cv2.VideoCapture("video_path.mp4")

# Video yazıcısını oluşturan yardımcı fonksiyon
# Giriş videosunun özelliklerine göre çıktı videosu için ayarları yapılandırır
def create_video_writer(video_cap, output_path):
    width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_cap.get(cv2.CAP_PROP_FPS))

    # codec değişikliği
    if cv2.VideoWriter_fourcc(*'mp4v') is not None:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    else:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output_path = output_path.replace('.mp4', '.avi')

    return cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# İki sınır kutusu arasındaki örtüşmeyi hesaplayan fonksiyon (Intersection over Union)
# Bu değer, aynı nesnenin farklı frame'lerdeki tespitlerini eşleştirmek için kullanılır
def calculate_iou(box1, box2):
    # box format: [x, y, w, h]
    box1_x1 = box1[0]
    box1_y1 = box1[1]
    box1_x2 = box1[0] + box1[2]
    box1_y2 = box1[1] + box1[3]

    box2_x1 = box2[0]
    box2_y1 = box2[1]
    box2_x2 = box2[0] + box2[2]
    box2_y2 = box2[1] + box2[3]

    # Kesişim alanını hesapla
    xi1 = max(box1_x1, box2_x1)
    yi1 = max(box1_y1, box2_y1)
    xi2 = min(box1_x2, box2_x2)
    yi2 = min(box1_y2, box2_y2)

    intersection_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    # Her iki kutunun alanını hesapla
    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)

    # Birleşim alanını hesapla
    union_area = box1_area + box2_area - intersection_area

    # IoU hesapla
    if union_area == 0:
        return 0

    iou = intersection_area / union_area
    return iou

# Video yazıcısını başlat
writer = create_video_writer(video_cap, "output.mp4")

# Eğitilmiş YOLO modelini yükle
# Bu model voleybol oyuncularını tespit etmek için kullanılacak
model = YOLO("model.pt")

# DeepSORT takip sistemini başlat
# max_age parametresi bir nesnenin kaç frame boyunca takip edileceğini belirler
tracker = DeepSort(max_age=30)

frame_count = 0

# Ana video işleme döngüsü
while True:
    # Her frame'in işlenme süresini ölç (FPS hesabı için)
    start = datetime.datetime.now()

    # Video'dan bir frame oku
    ret, frame = video_cap.read()
    if not ret:
        break

    # YOLO ile oyuncuları tespit et
    results = model(frame)[0]

    # Tespit edilen nesneleri saklamak için liste
    detections = []

    # Her tespit edilen nesne için gerekli bilgileri topla
    for data in results.boxes.data.tolist():
        # Güven skorunu al
        confidence = data[4]

        # Güven eşiğinin altındaki tespitleri atla
        if float(confidence) < CONFIDENCE_THRESHOLD:
            continue

        # Sınır kutusu ve sınıf kimliğini al
        xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
        class_id = int(data[5])

        # Tespitleri listeye ekle
        detections.append([[xmin, ymin, xmax - xmin, ymax - ymin], confidence, class_id])

    # DeepSORT ile oyuncuları takip et
    # Bu aşamada her oyuncuya unique bir ID atanır
    tracks = tracker.update_tracks(detections, frame=frame)

    # Takip edilen her oyuncu için görsel öğeleri ekle
    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        ltrb = track.to_ltrb()

        # Takip edilen nesnenin koordinatlarını al
        xmin, ymin, xmax, ymax = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])

        # Oyuncunun sınıfını (kimliğini) belirle
        # Eğer doğrudan tespit edilemezse, en yakın eşleşen tespiti kullan
        if hasattr(track, 'det_class'):
            class_id = track.det_class
        else:
            # En yakın detection'ı bul
            best_iou = 0
            best_class_id = None
            track_bbox = [xmin, ymin, xmax - xmin, ymax - ymin]

            for detection in detections:
                det_bbox = detection[0]
                det_class_id = detection[2]

                # IoU hesapla
                iou = calculate_iou(track_bbox, det_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_class_id = det_class_id

            class_id = best_class_id

        # İsmi al
        label = class_labels.get(class_id, "Unknown")

        # Oyuncunun etrafına dikdörtgen çiz ve ismini göster
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), GREEN, 2)

        # Etiketi ekrana yazdır
        cv2.rectangle(frame, (xmin, ymin - 20), (xmin + 150, ymin), GREEN, -1)
        cv2.putText(frame, f"{label} ({track_id})", (xmin + 5, ymin - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)

    # FPS hesapla ve ekranda göster
    end = datetime.datetime.now()
    total = (end - start).total_seconds()

    fps = f"FPS: {1 / total:.2f}"
    cv2.putText(frame, fps, (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 8)

    # İşlenmiş frame'i göster ve kaydet
    cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Frame", 1920, 1080)
    cv2.imshow("Frame", frame)
    writer.write(frame)
    if cv2.waitKey(1) == ord("q"):
        break

# İşlem bittiğinde kaynakları temizle
video_cap.release()
writer.release()
cv2.destroyAllWindows()