<img width="1280" height="720" alt="Gravity Forge 25 11 2026 21_06_51" src="https://github.com/user-attachments/assets/a1d6000f-64f3-4f90-a6d8-66a6e2ff82d6" />
<img width="1280" height="720" alt="Gravity Forge 25 11 2026 21_06_27" src="https://github.com/user-attachments/assets/95b80071-9281-4cc1-be21-0bbcdfbb1a83" />
# Gravity Forge

https://github.com/user-attachments/assets/5f88f366-6104-488a-b316-73f262a6b10e








3D N-body симулятор галактик и планетных систем на Python, NumPy и Pygame.

## Запуск из исходников

bash
pip install -r requirements.txt
python make_obj.py
python main.py

## Управление

- **ЛКМ (зажать)** — вращение камеры, **Колесо** — зум
- **ЛКМ (клик по объекту)** — привязка камеры, **F** — открепить, **[ / ]** — переключение объектов
- **ПКМ (клик)** — создать объект на орбите, **ПКМ + тянуть** — запуск по траектории
- **Q / E** — выбор объекта, **1–5** — сценарии
- **- / +** — скорость времени (0.1x–4x), **Пробел** — пауза
- **Tab** — режим ASCII, **O** — орбиты, **H** — скрыть HUD, **F11** — полный экран

# Features

- **Live orbit prediction & impact FX:** real-time dashed trajectory preview with collision marker when aiming (`RMB`), plus shockwave flashes and gravitational debris on body impacts
