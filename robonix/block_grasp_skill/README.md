# block_grasp_skill — 接入 Robonix

把 `block_grasp/` 的 YOLO-OBB + D1 灵巧手抓取管线薄封装成一个 Robonix Skill，
让 Pilot 大模型能用自然语言（`rbnx chat`）触发抓取。做法参照
`~/roboarm/robonix/roboarm_skill`。

## 架构

```
rbnx chat (Pilot, VLM)
    ↓ MCP
block_grasp skill node (FastMCP)      ← robonix/block_grasp_skill/block_grasp_skill/node.py
    ↓ Python import
block_grasp.grasp_controller.BlockGraspController  ← 原抓取管线（新增 grasp_once/stack_once）
    ↓
D1 机械臂 + 灵巧手 + RealSense
```

skill node 是薄层：只做 Robonix 生命周期 + 结果 JSON 化。检测/坐标/IK/抓取逻辑全部复用
`block_grasp/`。为支持无头调用，`grasp_controller.py` 新增了 `grasp_once` / `stack_once`
/ `reset_stack` / `move_home`（`run_loop` 的 CLI 行为不变）。

## MCP 工具

| 工具 | 作用 |
|------|------|
| `grasp_block(class_name="", position="")` | 按颜色抓一个积木放到 `position`（命名位置"中间"/颜色名 或 坐标"x,y"）；留空放到该颜色对应位置 |
| `stack_blocks(mover_class="", base_class="")` | 堆叠一次；给颜色对则把 mover 叠到 base 上，留空则底座=离 STACK_POSITION 最近者 |
| `reset_stack()` | 重置堆叠状态 |
| `move_home()` | 张手 + 机械臂回安全位 |

## 前置：Python 3.10 环境（必做一次）

默认 `bb_d1` 是 Python 3.8，跑不了 `robonix_api`/`mcp`（要 ≥3.10）。D1 SDK 有 cp310 wheel，
所以建一个 3.10 环境同时装齐两边依赖：

```bash
bash ~/Beingbeyond_D1/robonix/env_setup.sh      # 建 bb_d1_robonix + 装依赖（需联网）
```

## 构建 + 启动 + 验证

```bash
# 1) VLM 凭据（Pilot 用）
export VLM_BASE_URL=...   VLM_API_KEY=...   VLM_MODEL=...

# 2) 构建 skill（codegen 生成 Driver 生命周期 gRPC 桩）
cd ~/Beingbeyond_D1/robonix/block_grasp_skill
rbnx build -p .

# 3) 启动整套（atlas + executor + pilot + skill）
cd ../block_grasp_deploy
rbnx boot

# 4) 另开终端验证
rbnx caps                 # 应看到 block_grasp 技能已注册
rbnx tools                # 应看到 grasp_block / stack_blocks / reset_stack / move_home
rbnx chat                 # 试 "抓取红色积木" / "把两个积木叠起来"
```

> ⚠️ 安全：真机会实际运动。首次运行确认急停在手边、工作范围无人无障碍、留足观察空间。
> 首次调用工具时才初始化硬件（开相机/串口/CAN、加载 YOLO、移到安全姿态），会有几秒到十几秒延迟。

## 常见问题

- `ModuleNotFoundError: mcp / fastmcp / grpc_tools` — 3.10 环境依赖没装齐，重跑 `env_setup.sh`。
- `No module named beingbeyond_d1_sdk` — 3.10 环境没装 cp310 wheel（`env_setup.sh` 会装）。
- 找不到标定/权重 — 确认 `block_grasp/handeye_calib.npz` 与 `object_detect/runs/*/weights/best.pt` 存在。
- 机械臂/相机连接失败 — 检查 `BLOCK_GRASP_ARM_DEV`、`BLOCK_GRASP_HAND_CAN`、相机连接。
