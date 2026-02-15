## Ziel
Zuordnung einer Zug-Detektion zu einem Gleis (`G1`, `G2`, …).

## Pipeline
1. **MultiNormalizeTool**
   - berechnet Homographie `H`
   - warped Bild auf feste Canvas
2. **map_tool.py**
   - Gleise als Polyline zeichnen
   - erzeugt Band-Polygone (`trackmap.json`)
3. **Runtime**
   - skaliert Bild identisch
   - warped mit `H`
   - ordnet OBB per Overlap einem Gleis zu

## Methode
- Gleis = Band-Polygon → Maske  
- Zug = OBB-Polygon → Maske  
- größter Pixel-Overlap ⇒ Gleis-ID

## Wichtig
- `INPUT_SCALE` **muss** identisch zum Normalize-Tool sein
- Detektion im **warped Koordinatensystem**

## Output
```text
track_id = "G3"
overlap  = 1245 px
