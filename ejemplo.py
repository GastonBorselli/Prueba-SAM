import cv2

def capture_frames(video_path, output_folder, frame_interval=1):
    # Abrir el video
    cap = cv2.VideoCapture(video_path)

    # Comprobar si el video se abrió correctamente
    if not cap.isOpened():
        print("Error al abrir el video.")
        return

    # Contador para dar nombres a las imágenes capturadas
    frame_count = 0
    picNumber = 0
    # Leer y capturar los fotogramas del video
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Capturar la imagen cada cierto intervalo de fotogramas (frame_interval)
        if frame_count % frame_interval == 0:
            output_path = f"{output_folder}/{picNumber}.png"
            cv2.imwrite(output_path, frame)
            picNumber+=1
            print(f"Capturada imagen {output_path}")
		
        frame_count += 1
        

    # Cerrar el video y liberar recursos
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Ruta del video de entrada
    video_path = "videos/silla2.mp4"

    # Carpeta de salida para guardar las imágenes capturadas
    output_folder = "fotosExtraidas"

    # Intervalo de fotogramas para capturar (1 captura cada 'frame_interval' fotogramas)
    frame_interval = 20  # Capturará una imagen cada 30 fotogramas (aproximadamente 1 imagen por segundo)

    # Capturar imágenes del video y guardarlas en la carpeta de salida
    capture_frames(video_path, output_folder, frame_interval)