# Notebooks

Đặt tên theo thứ tự: `01_eda_v0.ipynb`, `02_eda_v1.ipynb`, `03_feature_ideas.ipynb`...

Notebook chỉ dùng để khám phá. Code nào dùng lại được thì chuyển vào `src/spike_ai/`.
Ở đầu notebook:

```python
from spike_ai.data import load_dirs
from spike_ai.features import build

df = load_dirs(["data/raw/v0"])
X, y = build(df, "v0", agent="opponent")
```
