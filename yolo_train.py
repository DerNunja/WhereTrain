from ultralytics import YOLO

def main():
    # Lade dein Modell
    model = YOLO("yolo11m.pt")

    # Trainiere das Modell
    results = model.train(
        data=r"C:\Users\Zayd Maatouf\Desktop\zug\train.yaml",  # <- Pfad zu deinem YAML
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,
        workers=4,
        patience=20,
        lr0=0.001,
        cache=True,
        project="train_runs",
        name="yolo11m_schilder",
        exist_ok=True,
        seed=42,
        deterministic=True,
        save=True,
        plots=True,
    )

    # Validation nach dem Training
    metrics = model.val()
    print(metrics)

# Multiprocessing-Schutz für Windows
if __name__ == "__main__":
    main()