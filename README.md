# Personenerkennung

Dieses Repository bietet eine verbesserte Version des Codes für die Personenerkennung aus dem `wise2122/soc-nn_main_repo` Repository.

### Benutzter Datensatz

Der Datensatz unter `data/custom_dataset` stammt aus `custom_dataset.zip` aus dem Repository `wise2122/personen-datensatz`.
Er enthält eine händisch ausgewählte Teilmenge der Bilder aus dem [COCO Datensatz](https://cocodataset.org), welcher unter der [Creative Commons Attribution 4.0 License](https://creativecommons.org/licenses/by/4.0) veröffentlicht wurde.

### Benutzter lizensierter Code

In `src/models` befindet sich lizensierter Code von [Xilinx Brevitas](https://github.com/Xilinx/brevitas), welcher unter der [3-Clause BSD License](https://opensource.org/license/BSD-3-clause) veröffentlicht wurde.

## Entwicklung

Bitte benutze [diesen Git Filter](https://gist.github.com/33eyes/431e3d432f73371509d176d0dfb95b6e) um Jupyter Notebook Dateien zu aufzubereiten bevor du sie pushst (z. B. Outputs und Metadaten entfernen).

Benutze `pip install -r requirements.txt` um die richtigen Versionen der benötigten Python-Pakete zu installieren.

### Entwicklung mit [Nix Flakes](https://wiki.nixos.org/wiki/Flakes)

Wenn du den [Nix Package Manager](https://nixos.org) benutzt, kannst du hiermit deine Entwicklungsumgebung einrichten. Ansonsten können die folgenden Schritte ignoriert werden.

```sh
# Development Shell starten
nix develop
# Virtual Environment erstellen (--copies ist wichtig für fix-python später)
python -m venv .venv --copies
# Virtual Environment aktivieren, z. B. für Fish Shell:
source .venv/bin/activate.fish
# Requirements installieren
pip install -r requirements.txt
# Virtual Environment für NixOS fixen
nix run github:GuillaumeDesforges/fix-python -- --venv .venv
```
