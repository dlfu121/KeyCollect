# 注意事项

Created by Ziyue 8/21/2026

## 一、每次重新打开终端，先执行

```bash
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate keycollect
cd "$HOME/dongziyue/KeyCollect"
unset PYTHONPATH
```

然后

```bash
python scripts/teleop.py assets/scenes/rm65_dexhand_scene.xml
```

如果只想看场景

```bash
python scripts/viewer.py assets/scenes/rm65_dexhand_scene.xml
```