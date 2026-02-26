# Umsetzung Gleis - Mapping
Hier kann man sehen wie das Mapping umgesetzt werden soll. Die Marker definieren verschiedene Bereiche (hier beispielhaft benannt "Abschnitt 1"). Mithilfe dieser Marker erstellen wir ein Normalisiertes Bild. Hierauf wird anschließend die Objekterkennung der Züge angewendet
![Normalisiertes Bild](Pictures/Normalized.png)
[^1]

Manuell Mappen wir dann die Gleise für diese Bereiche. So dass auch wir auch mit anderen Kamerapostionen und Orientierungen immer wissen wo die Gleise sich befinden.
![Mapping Prozess Darstellung](Pictures/Beispiel.png)
[^2]

Nach der [[Objekterkennung der Züge]] wird die Gleiskarte auf das normalisierte Bild projiziert. Für jede erkannte Zugdetektion wird geprüft, mit welchem Gleis-Polygon die größte Überlappung besteht. Der Zug wird anschließend diesem Gleis zugeordnet.
![Gleis Zuordnung](Pictures/Overlay.png)
[^3]
## Was konkret ist der Mapping Prozess?
Der Mapping Prozess fängt damit an, dass wir die einzelnen Bereiche (Gleise) mit Markern markieren. Dabei ist es vorteilhaft wenn alle Marker die gleiche Ausrichtung haben.

Dann werden mit dem ´section_tool.py´ die Abschnitte festgelegt z.B "Bahnhof". Abschnitte können einfach durch das anklicken der Marker erstellt werden. Diese Bereiche werden über ihre Marker definiert (das Tool ermittelt automatisch welche Marker verwendet werden. Sollten Marker verwendet werden welche das Tool nicht kennt müssen diese dem _DICT_CANDIDATES_ hinzugefügt werden). Dabei sollte man sich auf eine Markierungsstratigie einigen, z. B. TL/TR als „außen“ und BL/BR als „innen“.. 
![section_tool](Pictures/section_tool.png)
Wir verwenden zunächst: Top = außen, Bottom = innen (mit links/rechts entsprechend), sodass ein „entrolltes“ Schienennetz entsteht. Nachdem ein Bereich festgelegt wurde, wird er normalisiert (perspektivisch entzerrt) und als PNG gespeichert. Optional können in einem Bild mehrere Abschnitte definiert werden.
Alle relevanten Informationen (Section-ID, Marker-IDs, Canvas-Größe und Dictionary) werden direkt im Dateinamen des normalisierten Bildes abgelegt.

´´´bash
abschnitt_1__ids=TL1_TR7_BR5_BL2__1280x640__dict=DICT_ARUCO_ORIGINAL.png
´´´

- abschnitt_1 → Section-ID
- ids=TL1_TR7_BR5_BL2 → verwendete Marker-IDs pro Ecke
- 1280x640 → Canvas-Größe des normalisierten Bereichs
- dict=... → verwendetes ArUco-Dictionary

Der nächste Schritt ist das Mappen der Gleise. Dafür wird das ´map_tool.py´ verwendet. Hier zeichnet man die Gleise auf den normalisierten Abschnitten ein. Ein Gleis besteht dabei aus einer Polygon-Linie (möglichst mittig platzieren) und einem "Band" welches ungefähr so Breit wie die Schienen sein sollte. Die Polyline repräsentiert die geometrische Gleisachse. Das Band ist ein daraus generiertes Polygon, das die effektive Gleisbreite approximiert. Das Band wird für die Überlappungsberechnung verwendet, die Polyline für die Positionsprojektion entlang des Gleises. Das Tool gibt am Ende eine .json Datei aus mit den Gleisen und deren Beschreibung(track_id, polyline, band).
![map_tool](Pictures/map_tool.png)

## Kurzanleitung der Mapping-Tools

### 1) `section_tool.py` (Abschnitte normalisieren)

Zweck: Marker-basiert einen oder mehrere Abschnitte auswählen und perspektivisch entzerren.

**Ablauf**
1. Eingabebild in `IMAGE_PATH` setzen (oder Webcam aktivieren).
2. Tool starten und Dictionary automatisch erkennen lassen.
3. Abschnitt anlegen (`A`) und Canvas-Preset wählen (`1..6`).
4. Marker-Ecken für den aktiven Abschnitt setzen (`1=TL`, `2=TR`, `3=BR`, `4=BL`, danach Klick auf Marker).
5. Abschnitt exportieren (`X`) oder alle Abschnitte exportieren (`E`).

**Wichtige Ausgaben**
- Normalisierte Abschnittsbilder im Ordner `Mapping/Sections/`.
- Dateiname enthält `section_id`, Marker-IDs und Canvas-Größe, z. B.:

```text
abschnitt_1__ids=TL1_TR7_BR5_BL2__1280x640__dict=DICT_ARUCO_ORIGINAL.png
```

### 2) `map_tool.py` (Gleise mappen)

Zweck: Auf einem normalisierten Abschnittsbild Gleise und optionale Zonen einzeichnen.

**Ablauf**
1. `IMAGE_PATH` auf ein exportiertes Abschnittsbild setzen.
2. In `track`-Mode Punkte für eine Gleis-Mittellinie klicken.
3. Mit `ENTER` speichern (Band wird aus Linienbreite automatisch erzeugt).
4. Optional weitere Gleise oder Zonen (`Z`) einzeichnen.
5. Mapping mit `S` als `__trackmap.json` speichern.

**Wichtige Shortcuts**
- `T`/`Z`: Track-/Zone-Modus
- `N`: aktuelle Eingabe löschen
- `BACKSPACE`: letzten Punkt entfernen
- `+`/`-`: Bandbreite ändern
- `.` / `,`: nächstes/vorheriges Gleis
- `S`: JSON speichern
- `Q`/`ESC`: beenden


### Offene Frage
**Wie wird das Bild am besten normalisiert?**
Sollte alles auf ein festes x:y Format normalisiert werden oder sollten die Größen der einzelnen Bereiche mit in Betracht gezogen werden? Also längliche Bereiche in ein eher längliches Format und quadratische Bereich in ein entsprechend quadratisches Format.
#### Spezifische Seitenverhältnisse

| Vorteile                                    | Nachteile                            |
| ------------------------------------------- | ------------------------------------ |
| Geometrisch „schöner“                       | Unterschiedliche Modell-Inputs       |
| Gleise behalten ihr natürliches Verhältnis  | Komplexere Pipeline                  |
| Projektion auf Gleisachsen minimal sauberer | Unterschiedliche Gleiskarten-Formate |
|                                             | Fusion & Tracking schwerer           |
#### Einheitliches Format (bevorzugt)
Diese Variante wäre leichter um zusetzen 

| Vorteile                                     | Nachteile                                       |
| -------------------------------------------- | ----------------------------------------------- |
| Einfachere Pipeline                          | Verzerrung bei sehr länglichen Bereichen        |
| Einheitliche Modell-Inputs                   | Gleise können „gestaucht“ oder „gezogen“ werden |
| Gleiskarten lassen sich **wiederverwenden**  |                                                 |
| Tracking & Fusion zwischen Kameras einfacher |                                                 |
| Weniger Spezialfälle im Code                 |                                                 |
Mann könnte für gewisse Bereiche Formate festlegen um so starke Verzerrung vorzubeugen.
Wir verwenden **wenige Standardformate** (z. B. 2:1 und 1:1) je nach Abschnittstyp. Das wäre noch relativ leicht umzusetzen. Mit reinspielen würde da wahrscheinlich auch eine kluge Marker Platzierung. 

[^1]: Normalisiertes Bild

[^2]: Mapping Prozess Darstellung

[^3]: Gleis Zuordnung
