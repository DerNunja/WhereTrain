## Zug Erkennung mit Farben
detection_with_color.py: Implementiert eine Funktion welche grüne und gelbe Pixel in einem Bild erkennt und die größte zusammenhängenden Pixelregion als binäre Maske erzeugt. Diese Maske wird dann verwendet um eine Bounding Box zu berechnen, welche den Bereich der erkannten Farben einschließt.

Die Funktion ist sehr basic und könnte um einiges verbessert werden. Züge sollten eher länglich sein daher wäre es z.B praktisch wenn nur längliche BBoxen weitergeben werden. Am besten wäre es wahrscheinlich wenn man nach farbigen Pixeln nur innerhalb der Gleise sucht.

Aber da die Funktion nur da war um ein erstes Prototyp zu haben, ist sie sehr einfach gehalten.