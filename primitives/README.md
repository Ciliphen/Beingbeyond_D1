# D1 原语（Primitives）

BeingBeyond D1 机器人的 Robonix 原语包。

## 原语包

| 包 | 命名空间 | 硬件 | 能力约定数 |
|------|-----------|------|:---------:|
| `d1_hand` | `beingbeyond/primitive/hand/*` | DexHand 灵巧手（6-DOF, CAN） | 7 |
| `d1_arm` | `beingbeyond/primitive/arm/*` | HeadArmRobot 机械臂（8-DOF, 串口） | 4 |
| `d1_camera` | `beingbeyond/primitive/camera/*` | Intel RealSense D435i 深度相机（USB） | 8 |

## 架构

```
skill（VLA 策略、LLM 任务）  ← "抓取方块"
  │
service（规划、场景）        ← "目标位姿 = (x, y, z)"
  │
primitive（硬件驱动）        ← "arm.move_joint([0.5, ...])"
  │
硬件（CAN / 串口 / USB）
```

每个原语是独立 Python 进程，遵循严格生命周期：
`REGISTERED → INACTIVE → ACTIVE → ERROR → TERMINATED`

## d1_hand — 灵巧手

D1 欠驱动灵巧手（6 关节，CAN 通信）。

| 能力约定 | 模式 | MCP | 说明 |
|----------|------|:---:|------|
| `beingbeyond/primitive/hand/driver` | rpc | — | 生命周期 |
| `beingbeyond/primitive/hand/move_joint` | rpc | ✅ | 设置各关节目标角度（0=张开, 1=闭合） |
| `beingbeyond/primitive/hand/state_joint` | topic_out | — | 读取当前关节角度 |
| `beingbeyond/primitive/hand/state_speed` | topic_out | — | 读取当前关节速度 |
| `beingbeyond/primitive/hand/set_speed` | rpc | — | 设置各关节速度上限 |
| `beingbeyond/primitive/hand/set_torque` | rpc | — | 设置各关节力矩上限 |
| `beingbeyond/primitive/hand/info` | rpc | ✅ | 查询关节名称、数量、型号 |

部署：`rbnx start -p primitives/d1_hand -s hand_type=right -s can_iface=can0`

## d1_arm — 机械臂

D1 机械臂（8 关节，串口通信）。

| 能力约定 | 模式 | MCP | 说明 |
|----------|------|:---:|------|
| `beingbeyond/primitive/arm/driver` | rpc | — | 生命周期 |
| `beingbeyond/primitive/arm/move_joint` | rpc | ✅ | 设置各关节绝对角度（弧度） |
| `beingbeyond/primitive/arm/state_joint` | topic_out | — | 读取当前关节角度 |
| `beingbeyond/primitive/arm/info` | rpc | ✅ | 查询关节名称、数量、型号 |

部署：`rbnx start -p primitives/d1_arm -s dev=/dev/ttyUSB0 -s urdf_path=<路径>`

## d1_camera — 深度相机

Intel RealSense D435i RGB-D 相机。

| 能力约定 | 模式 | MCP | 说明 |
|----------|------|:---:|------|
| `beingbeyond/primitive/camera/driver` | rpc | — | 生命周期 |
| `beingbeyond/primitive/camera/info` | rpc | ✅ | 查询型号、分辨率、帧率 |
| `beingbeyond/primitive/camera/rgb` | topic_out | — | 连续 RGB 图像流（高频） |
| `beingbeyond/primitive/camera/depth` | topic_out | — | 连续深度图像流（高频） |
| `beingbeyond/primitive/camera/snapshot` | rpc | ✅ | 按需取一帧 RGB（LLM 调用） |
| `beingbeyond/primitive/camera/depth_snapshot` | rpc | ✅ | 按需取一帧深度图 |
| `beingbeyond/primitive/camera/rgbd` | rpc | — | 同步 RGB-D 图像对 |
| `beingbeyond/primitive/camera/intrinsics` | rpc | — | 相机内参（fx, fy, cx, cy, 畸变） |

部署：`rbnx start -p primitives/d1_camera -s width=640 -s height=480 -s fps=30`

## VLA 模型对接

Being-H VLA 模型输出 12D 绝对关节位置，直接对接两条原语：

```
模型输出                               原语调用
─────────────────────────────────     ───────────────────────────
action.arm_joint_position（6D, rad） → arm.move_joint(positions)
action.dexhand_position（6D, 0-1） → hand.move_joint(positions)
```

D1 推理客户端工作流：

1. `camera.snapshot()` → 获取图像，输入 VLA 模型
2. `arm.state_joint()` + `hand.state_joint()` → 获取状态，输入 VLA 模型
3. 将（图像, 状态）发送给 VLA 推理服务器 → 获取动作
4. `arm.move_joint()` + `hand.move_joint()` → 执行动作

## 设计文档

- [飞书 — 原语设计](https://vcnx4dozoypf.feishu.cn/wiki/KHQkwullsids0SkBxWOcHTERnle)
- [飞书 — 灵巧手项目](https://vcnx4dozoypf.feishu.cn/docx/ThShdPZployNwSxIUXHcX0hAnlh)
- [Robonix — 原语接口目录](https://github.com/syswonder/robonix/tree/dev/rust/robonix-interfaces)
