# Cybercoon Setup

Bu klasor icin kurulum adimlari asagidadir.

## 1. Virtual environment olustur

Proje kok dizinindeyken:

```powershell
python -m venv .venv
```

## 2. Venv aktif et

PowerShell icin:

```powershell
.\.venv\Scripts\Activate.ps1
```

CMD icin:

```bat
.venv\Scripts\activate.bat
```

## 3. Requirements kur

Strateji klasorune gecip paketleri yukle:

```powershell
cd strategies\cybercoon
python -m pip install -r requirements.txt
```

## 4. Stratejiyi calistir

```powershell
python strategy.py
```

## Notlar

- `strategy.py` ana calisma dosyasidir.
