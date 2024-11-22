from ultralytics import YOLO
import cv2
import os

# Cargar el modelo YOLOv8 preentrenado
model = YOLO('yolov8n.pt')  # Puedes cambiar a 'yolov8s.pt', 'yolov8m.pt', etc., según tus necesidades

# Inicializa la cámara
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("No se pudo acceder a la cámara.")
    exit()

# Crear carpeta para las capturas si no existe
os.makedirs("screenshots", exist_ok=True)
screenshot_count = 0

# Abre un archivo para guardar las detecciones (se abre una sola vez)
with open("detections.txt", "w") as f:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error al capturar el video.")
            break

        # Realizar detecciones
        results = model(frame)

        # Guardar las detecciones en el archivo
        for result in results:
            for box in result.boxes:
                class_name = model.names[int(box.cls)]  # Nombre de la clase
                confidence = box.conf.item()  # Convierte la confianza a un número Python
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # Convierte el tensor a lista y luego a enteros
                f.write(f"{class_name} {confidence:.2f} {x1} {y1} {x2} {y2}\n")

        # Mostrar el frame con las detecciones
        annotated_frame = results[0].plot()
        cv2.imshow("Detección en tiempo real", annotated_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):  # Salir con 'q'
            break
        elif key == ord(' '):  # Barra espaciadora para guardar captura
            screenshot_path = f"screenshots/frame_{screenshot_count}.jpg"
            cv2.imwrite(screenshot_path, annotated_frame)
            print(f"Captura guardada en: {screenshot_path}")
            screenshot_count += 1

cap.release()
cv2.destroyAllWindows()
