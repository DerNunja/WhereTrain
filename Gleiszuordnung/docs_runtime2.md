#  Track Assignment & Positioning System

## ## Ziel
Automatisierte Zuordnung von Zug-Detektionen zu Gleisen (G1, G2, ...) und präzise Positionsbestimmung entlang der Gleismittellinie (Anfang → Ende).

---

## ## Pipeline
1. **MultiNormalizeTool** (Optional/Vorbereitung)
   * Berechnet Homographie $H$.
   * Warpt das Kamerabild auf eine feste Canvas (z.B. 1280×640).
2. **map_tool.py** (Konfiguration)
   * Zeichnet Gleise als **Polylines** (Mittellinie).
   * Erzeugt **Band-Polygone** (Gleisbreite) für die Flächenerkennung.
   * Speichert alle Geometrien in der `trackmap.json`.
3. **Runtime** (Inferenz)
   * Skaliert/Warpt Input-Bild identisch zur Canvas.
   * Empfängt OBB-Polygone (Oriented Bounding Boxes) von YOLO.
   * Ordnet jedes OBB per Pixel-Overlap einem Gleis zu.

---

## ## Methode
* **Gleiszuordnung:** * Gleis = Band-Polygon → Maske.
  * Zug = OBB-Polygon → Maske.
  * Entscheidung via: $argmax(Overlap_{px})$.
  * Falls `overlap < MIN_OVERLAP_PX` -> Kein gültiges Gleis.
* **Positionsberechnung:**
  * Berechnung des Schwerpunkts (Centroid) des Zug-Polygons.
  * Projektion des Punktes auf die Gleis-Mittellinie.
  * Bestimmung der zurückgelegten Strecke entlang der Polyline.

---

## ## Wichtig
* **INPUT_SCALE:** Muss absolut identisch zum Normalize-Tool sein (1280×640).
* **Koordinaten:** Detektionen müssen im **Warped-System** (Canvas) vorliegen.
* **Richtung:** Die Reihenfolge der Punkte in der Polyline definiert den Startpunkt ($s_{norm} = 0.0$).

---

## ## Output (Beispiel)
```text
track_id     = "G3"          // Identifiziertes Gleis
overlap      = 1245 px       // Fläche der Übereinstimmung
s_px         = 842.3 px      // Distanz in Pixeln ab Start
s_norm       = 0.63          // 63% Position (Anfang -> Ende)
s_norm_rev   = 0.37          // 37% Position (Ende -> Anfang)
lateral_px   = 5.2 px        // Seitlicher Versatz zur Mitte