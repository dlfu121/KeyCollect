# KeyCollect 交付清单

生成日期：2026-09-06（Asia/Shanghai）。本清单只记录仓库中实际可核验的工件；没有实测数据的结果统一标为待实测。

## 已交付

- 统一系统契约：`config/system_contract.yaml`，SHA-256 `6e1675c6be7f12d8f4c10a5c3ddad62318f311543740e175fe60a485e278ce19`。
- 受管动捕协议：`config/mocap_protocol_v1.yaml`，对时间戳、帧序号、单位、版本和被消费通道名作严格检查。
- 新 RGB-D 24 FPS 录制配置：`config/record_mujoco.yaml`；历史双 RGB 30 FPS 推理配置：`config/infer_act_rgb30.yaml`。
- 固定评测协议：`config/task_evaluation.yaml`；确定生成的 100 个 case：`artifacts/evaluation_cases.jsonl`。
- 重新生成的 MJCF：`assets/scenes/rm65_dexhand_scene.xml`，SHA-256 `090e5bcf38d3349151f9138af1779cba6b69057200e4998f5bf957aecbf1d8da`。
- 中间 URDF：`assets/scenes/rm65_dexhand_scene.urdf`，SHA-256 `5215120840e9bf295d340508230ff1ce1af14e19f7a779e8d190e8bc74ab6a87`。
- 依赖锁：`uv.lock`，SHA-256 `78df693b915246d266f9f2a27e7019aab1f08ca97a4d2d3117353ce546bae5a6`。
- 数据集完整性报告：`artifacts/dataset_audit/report.json`；127 行 episode 清单：`artifacts/dataset_audit/episodes.csv`；episode 级划分：`artifacts/dataset_audit/split.json`。
- 现有 40k checkpoint 训练记录缺失审计：`artifacts/act_training_manifest.json`。
- 独立评测状态：`artifacts/act_independent_evaluation.json`。
- 治理训练入口：`scripts/train_act_governed.py`，强制完整 episode 划分、排除测试集、记录日志/种子/硬件/依赖/耗时/checkpoint 哈希。
- 固定任务评测入口：`scripts/evaluate_act.py`，逐 case 保存视频、结果及运行时记录；不满足完整协议或独立资格时不会生成成功率。
- 16 项逐项状态：`artifacts/requirements_status.md`。
- 未完成项目：`artifacts/uncompleted_items.md`。

## 实际验证结果

- RM65 解析/FK/IK/连续选支：6/6 通过，完整输出为 `artifacts/test-results/rm65.log`。
- KeyCollect 集成测试：42/42 通过，完整输出为 `artifacts/test-results/integration.log`。
- 合计：48/48 通过，汇总为 `artifacts/test-results/summary.json`。
- 实际 40k ACT checkpoint 通过启动严格契约检查和一个无界面在线步骤，原始记录为 `artifacts/act_online_smoketest.jsonl`。该记录仅是接口冒烟，不是延迟基准或抓取成功率。
- 固定评测链完成一个故意缩短的 case 冒烟，结果为 `artifacts/act_evaluation_smoke/case_000/result.json`；其中 `success=null`、`status=smoke_incomplete`，不属于独立评测。

## 数据与评测结论

- 现有数据集实测审计为 127 条 episode、39,673 帧、30 FPS、39 维状态、26 维动作，完整性检查通过。
- 历史 episode 没有记录成功标签、失败原因、物体初始位姿、当时场景哈希、随机种子、采集批次和四阶段时间戳；清单已逐条标为 `UNAVAILABLE_NOT_RECORDED` 或 `PENDING_ANNOTATION`，未追溯伪造。
- 现有 40k checkpoint 使用全部 127 条 episode 训练，因此不具备独立测试资格。抓取成功率、峰值力、滑移率和在线延迟统计均为待实测。
- 真机失联响应、真实动捕侧摆、触觉标定、RGB-D/触觉数据与 checkpoint、完整导纳控制和固定任务完整运行均尚未完成，详见未完成事项。
