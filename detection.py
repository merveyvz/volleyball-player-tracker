# Gerekli kütüphaneleri içe aktar
from ultralytics import YOLO
import os

# Model eğitimi için kullanılan fonksiyon
# model_path: Eğitilecek başlangıç modelinin yolu
# data_yaml_path: Veri seti konfigürasyon dosyasının yolu
# epochs: Toplam eğitim tur sayısı
# imgsz: Görüntü boyutu
def train_model(model_path, data_yaml_path, epochs=30, imgsz=640):
    model = YOLO(model_path)
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=imgsz
    )
    return results

# Eğitilen modelin performansını değerlendiren fonksiyon
# Doğruluk oranı, hassasiyet gibi metrikleri hesaplar
def validate_model(model_path, data_yaml_path):
    model = YOLO(model_path)
    results = model.val(data=data_yaml_path)
    return results


# Eğitilen model ile yeni görüntüler üzerinde tahmin yapan fonksiyon
# conf: Güven eşiği - bu değerden düşük tahminler göz ardı edilir
def predict_images(model_path, source_path, conf=0.5):
    model = YOLO(model_path)
    results = model.predict(
        source=source_path,
        conf=conf,
        save=True
    )
    return results

if __name__ == "__main__":
    # Proje dizin yapısı için gerekli yolları ayarla
    base_path = r"Proje dizini yolu"  
    data_yaml = os.path.join(base_path, "data.yaml")  # Veri seti konfigürasyonu
    initial_model = "yolov8m.pt"  # Başlangıç modeli
    
    # Modeli eğit
    print("Model eğitimi başlıyor...")
    train_results = train_model(initial_model, data_yaml)
    
    # Eğitim sonrası en iyi model ağırlıklarının yolunu belirle
    best_weights = os.path.join(base_path, "runs", "detect", "train", "weights", "best.pt")
    
    # Modeli değerlendir
    # Bu aşama modelin ne kadar iyi çalıştığını gösterir
    print("Model değerlendirmesi yapılıyor...")
    val_results = validate_model(best_weights, data_yaml)
    
    # Test görüntüleri üzerinde tahmin yap
    # Sonuçlar görsel olarak kaydedilecek
    print("Test görüntüleri üzerinde tahminler yapılıyor...")
    test_images_path = os.path.join(base_path, "images", "test")
    pred_results = predict_images(best_weights, test_images_path)

