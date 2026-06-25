# D1 Primitives

Robonix primitive packages for the BeingBeyond D1 robot.

## d1_hand

D1 Dexterous Hand primitive — wraps the D1 DexHand SDK (6-DOF underactuated hand
over CAN) as a Robonix capability provider.

### Contract surface

| Contract ID | Mode | Transport | MCP | Description |
|---|---|---|---|---|
| `beingbeyond/primitive/hand/driver` | rpc | gRPC | — | Lifecycle entry point |
| `beingbeyond/primitive/hand/move_joint` | rpc | MCP | ✅ | Set ND joint positions (`float64[]`, 0=open/1=closed) |
| `beingbeyond/primitive/hand/state_joint` | topic_out | ROS2 | — | Read joint positions |
| `beingbeyond/primitive/hand/state_speed` | topic_out | ROS2 | — | Read joint speeds |
| `beingbeyond/primitive/hand/set_speed` | rpc | gRPC | — | Set speed limit per joint |
| `beingbeyond/primitive/hand/set_torque` | rpc | gRPC | — | Set torque limit per joint |
| `beingbeyond/primitive/hand/info` | rpc | MCP | ✅ | Query joint names, count, model |

### Joint layout (D1)

| Index | Name | Finger |
|:---:|---|---|
| 0 | `thumb_cmc_pitch` | Thumb |
| 1 | `thumb_cmc_yaw` | Thumb |
| 2 | `index_mcp_pitch` | Index |
| 3 | `middle_mcp_pitch` | Middle |
| 4 | `ring_mcp_pitch` | Ring |
| 5 | `pinky_mcp_pitch` | Pinky |

Values in [0, 1]: 0 = open (straight), 1 = closed (bent).

### Quick test (standalone, no Robonix required)

```bash
conda activate bb_d1
python examples/test_hand_primitive.py
```

### Deploy with Robonix

```bash
rbnx start -p primitives/d1_hand \
  -s hand_type=right \
  -s can_iface=can0 \
  -s baudrate=1000000
```

### Design docs

Primitive interface design: [Feishu Wiki — 灵巧手项目 / 原语设计](https://vcnx4dozoypf.feishu.cn/docx/ThShdPZployNwSxIUXHcX0hAnlh)
