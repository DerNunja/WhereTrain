# Where Train?

Ziel dieses Projekts ist es, die Vorarbeit von **Paul Kellner** zur
kontextbezogenen Standortbestimmung von Modellzügen zu erweitern.
![Aktueller Stand und Erweiterung](Picture1.png)
Die Arbeit von Paul beantwortet die Frage:

> **„Befindet sich ein Zug innerhalb einer definierten Standortzone (rot)?“**

Unser Projekt erweitert diesen Ansatz wie folgt:

> In der Arbeit von Paul Kellner wird bestimmt, ob sich ein Zug innerhalb einer
> definierten Standortzone befindet. Aufbauend auf diesem Ansatz verfeinern wir
> die Lokalisierung, indem wir die Zone in einzelne Gleise (grün) unterteilen und
> zusätzlich die Position des Zuges entlang eines konkreten Gleises bestimmen (schwarzes Kreuz).
> Dadurch wird aus einer binären Anwesenheitsdetektion eine präzise,
> gleisbasierte Lokalisierung.

## Geleistete Vorarbeit

- Zugriff auf das GitLab-Repository von Paul Kellner ist vorgesehen  
  (aktuell noch Anmeldeprobleme, Zugriff sehr wahrscheinlich)
- Unabhängig vom Zugriff übernehmen wir **konzeptionell**:
  - Nutzung von **ArUco-Markern**
  - **Homographie-basierte Kamera-Normalisierung**
  - Arbeit auf einem **stabilen, normalisierten Kamerabild**

Diese Normalisierung bildet die Grundlage für alle weiteren Schritte.

## Unser Part

### Gleisbasierte Lokalisierung

- Arbeiten ausschließlich auf dem **normalisierten Bild**
- Modellierung jedes Gleises als:
  - schmale Polygonfläche oder *Polygonlinie*
- Detektierte Züge:
  - Bounding Boxes (Baseline)
  - optional: Segmentationsmasken (für höhere Genauigkeit)

**Zuordnung:**

- Für jede Detektion wird geprüft, **mit welchem Gleispolygon die größte
  Überlappung besteht**
- Der Zug wird diesem Gleis zugeordnet

```bash
Zug → Gleis_3
```

### Position auf dem Gleis

Zusätzlich zur Gleis-ID bestimmen wir die **Position entlang des Gleises**:

- Projektion der Zugposition auf die Gleisachse
- Darstellung als:
  - Pixelposition oder
  - normierter Wert (z. B. 0.0 – 1.0)

```bash
Zug → Gleis_3 → Position = 0.72
```

## runtime.py

Einmalig beim Start:

ArUco-Dictionary wird automatisch aus dem ersten Frame bestimmt und für alle folgenden Frames wiederverwendet (`autodetect_dictionary` läuft nur einmal).
Ein einzelner ArucoDetector wird erstellt und in dem gesamten Loop genutzt.

Pro Frame läuft dieser Ablauf:

1. Marker im Eingabebild erkennen (mit festem Dictionary).
2. Für jeden Abschnitt Homographie aus sichtbaren Marker-Zentren berechnen.
3. Falls Marker kurz fehlen: letzte Homographie pro Abschnitt bis `H_TIMEOUT_SEC` weiterverwenden (Cache in-memory).
4. Auf dem gewarpten Abschnitt Züge per `detect_by_color(...)` (Platzhalter für Yolo) detektieren.
5. Jede BBox über maximale Band-Überlappung einem Gleis zuordnen (`MIN_OVERLAP_PX` als Schwellwert).
6. Für zugeordnete Gleise Position entlang der Polyline (`s_norm`) und lateralen Abstand berechnen.

Wichtig: Der aktuelle Code implementiert keine zeitliche Glättung und kein Distanz-basiertes Tie-Breaking bei gleicher Overlap-Fläche; verwendet wird nur die größte Überlappung. Bekannte Performance-Engpässe liegen im Debug-Rendering (`draw_tracks_overlay`, mehrere imshow-Fenster) sowie in der farbbasierten Detektion — mit `SHOW_DEBUG = False` lässt sich der Durchsatz deutlich steigern.

## Repository-Überblick (Skripte)

| Skript | Zweck | Typischer Einsatz |
| --- | --- | --- |
| `runtime.py` | Führt die komplette Laufzeit-Pipeline aus: Marker erkennen, Abschnitt warpen, Zug detektieren, Detektion auf Gleis mappen, Position entlang des Gleises berechnen. | Online/Offline-Demo für den Gesamtablauf mit Webcam, Bild oder Video. |
| `extract_train_data.py` | Extrahiert normalisierte Trainingsbilder aus Videos (`data/TrainVid*.mp4`) und speichert nur Frames mit genug relevanten Farbanteilen. | Datensatzerstellung für spätere Modell-Trainingsläufe. |
| `Mapping/section_tool.py` | Interaktives Tool zum Definieren von Abschnitten über ArUco-Marker und Export normalisierter Abschnittsbilder. | Erster Schritt beim Setup neuer Kameraperspektiven. |
| `Mapping/map_tool.py` | Interaktives Tool zum Einzeichnen von Gleis-Polylinien und -Bändern auf normalisierten Abschnitten; Export als `__trackmap.json`. | Erstellen oder Pflegen der Gleiskarte pro Abschnitt. |
| `Detection/Color_detcion/detection_with_color.py` | Einfache farbbasierte Zugdetektion (HSV, Morphologie, größte Komponente -> BBox). | Prototyp/Baseline ohne trainiertes Modell. |
| `main.py` | Minimales Platzhalter-Entrypoint-Skript. | Aktuell ohne funktionale Pipeline-Relevanz. |

## Schneller Workflow (empfohlen)

1. Mit `Mapping/section_tool.py` Abschnitte definieren und normalisierte Bilder exportieren.
2. Mit `Mapping/map_tool.py` pro Abschnitt Gleise als Polyline + Band einzeichnen.
3. Mit `runtime.py` die Laufzeit-Pipeline auf Video/Webcam testen.
4. Optional mit `extract_train_data.py` zusätzliche normalisierte Trainingsbilder aus Rohvideos erzeugen.


## Mögliche Erweiterungen / Verbesserungen

### Aktuelle Einschränkung

- Das Schienennetz muss manuell im Bild modelliert werden
- Änderungen der Kameraposition oder Perspektive können die Genauigkeit beeinflussen (wie stark muss noch getestet werden)
- Sollten sich die Marker verschieben, funktioniert das Mapping der Gleise nicht mehr

### Erweiterungsidee: Automatische Schienenerkennung

Statt manueller Modellierung könnten die Schienen direkt aus dem Bild segmentiert werden.
> [!NOTE]
> Automatisierte Schienen Erkennung
> [Efficient railway track region segmentation algorithm based on lightweight neural network and cross-fusion decoder - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0926580523003291)

## Interessante/möglicherweise nützliche Quellen

Automatisierte Schienen Erkennung:
[Efficient railway track region segmentation algorithm based on lightweight neural network and cross-fusion decoder - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0926580523003291)

