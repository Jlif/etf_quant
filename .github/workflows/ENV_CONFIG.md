# GitHub Actions 环境变量配置指南

## 概述

GitHub Actions 支持多种方式配置环境变量，适用于不同的使用场景。本文档介绍如何为 ETF 量化项目的 workflow 配置环境变量。

## 配置方式对比

| 方式 | 作用范围 | 安全性 | 适用场景 |
|------|---------|--------|----------|
| **Repository Secrets** | 整个仓库 | 加密存储 | Webhook URL、API Key 等敏感信息 |
| **Repository Variables** | 整个仓库 | 明文存储 | 配置参数、开关等非敏感信息 |
| **Workflow 内联 env** | 单个 workflow | 明文存储 | 固定的常量配置 |
| **Step 级别 env** | 单个 step | 明文存储 | 某个步骤特有的配置 |

---

## 方式一：Repository Secrets（推荐用于敏感信息）

适用于：飞书 Webhook URL、API Key、密码等敏感信息

### 配置步骤

1. 进入 GitHub 仓库页面
2. 点击 `Settings` → `Secrets and variables` → `Actions`
3. 点击 `New repository secret`
4. 填写：
   - **Name**: `FEISHU_WEBHOOK_URL`
   - **Value**: `https://open.feishu.cn/open-apis/bot/v2/hook/xxxxxx`
5. 点击 `Add secret`

### 在 Workflow 中使用

```yaml
- name: Send Feishu webhook alert
  env:
    FEISHU_WEBHOOK: ${{ secrets.FEISHU_WEBHOOK_URL }}
  run: |
    curl -X POST "$FEISHU_WEBHOOK" \
      -H "Content-Type: application/json" \
      -d '{"msg_type": "text", "content": {"text": "test"}}'
```

### 常用 Secrets 清单

| Secret 名称 | 说明 | 示例值 |
|------------|------|--------|
| `FEISHU_WEBHOOK_URL` | 飞书群机器人 Webhook | `https://open.feishu.cn/open-apis/bot/v2/hook/xxx` |
| `WECOM_WEBHOOK_URL` | 企业微信群机器人 Webhook | `https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=xxx` |
| `DINGTALK_WEBHOOK_URL` | 钉钉群机器人 Webhook | `https://oapi.dingtalk.com/robot/send?access_token=xxx` |
| `AKSHARE_PROXY` | akshare 代理配置（可选） | `http://proxy.example.com:8080` |

---

## 方式二：Repository Variables（推荐用于非敏感配置）

适用于：策略参数、运行开关、通知配置等非敏感信息

### 配置步骤

1. 进入 GitHub 仓库页面
2. 点击 `Settings` → `Secrets and variables` → `Actions`
3. 切换到 `Variables` 标签页
4. 点击 `New repository variable`
5. 填写：
   - **Name**: `SIGNAL_STRATEGY`
   - **Value**: `动量轮动策略`
6. 点击 `Add variable`

### 在 Workflow 中使用

```yaml
- name: Run signal generation
  env:
    STRATEGY_NAME: ${{ vars.SIGNAL_STRATEGY }}
  run: |
    python latest_signal.py --config config.yaml --strategy "$STRATEGY_NAME"
```

### 常用 Variables 清单

| Variable 名称 | 说明 | 示例值 |
|-------------|------|--------|
| `SIGNAL_STRATEGY` | 指定运行的策略名称 | `动量轮动策略` |
| `CUTOFF_DATE` | 指定交易截止日 | `20260622` |
| `ALWAYS_NOTIFY` | 是否总是发送通知（无视调仓） | `false` |
| `NOTIFY_TIME` | 定时运行时间（cron 表达式） | `30 7 * * *` |

---

## 方式三：Workflow 级别环境变量

适用于：整个 workflow 共享的固定配置

```yaml
name: ETF Daily Signal Alert

env:
  # 策略配置
  STRATEGY_NAME: "动量轮动策略"
  CONFIG_PATH: "config.yaml"
  
  # 通知配置
  NOTIFY_ON_REBALANCE_ONLY: "true"
  
  # 数据源配置
  DATA_SOURCE_PROVIDER: "akshare"

on:
  schedule:
    - cron: '30 7 * * *'
  workflow_dispatch:

jobs:
  signal-alert:
    runs-on: ubuntu-latest
    steps:
      - name: Run signal generation
        run: |
          python latest_signal.py --config "$CONFIG_PATH" --strategy "$STRATEGY_NAME"
```

---

## 方式四：Job 级别环境变量

适用于：某个 job 特有的配置

```yaml
jobs:
  signal-alert:
    runs-on: ubuntu-latest
    env:
      # 这个 job 特有的环境变量
      BUFFER_DAYS: "10"
      LOOKBACK_DAYS: "22"
    steps:
      - name: Run signal
        run: |
          echo "使用 lookback: $LOOKBACK_DAYS, buffer: $BUFFER_DAYS"
```

---

## 方式五：Step 级别环境变量

适用于：某个 step 特有的配置，会覆盖上层同名变量

```yaml
- name: Run signal generation
  env:
    # 仅这个 step 使用
    PYTHONUNBUFFERED: "1"
    LOG_LEVEL: "INFO"
  run: |
    python latest_signal.py --config config.yaml
```

---

## 完整配置示例

结合多种方式的完整 workflow 示例：

```yaml
name: ETF Daily Signal Alert

# Workflow 级别环境变量
env:
  CONFIG_PATH: "config.yaml"
  NOTIFY_ON_REBALANCE_ONLY: "true"

on:
  schedule:
    - cron: '30 7 * * *'
  workflow_dispatch:

jobs:
  signal-alert:
    runs-on: ubuntu-latest
    
    # Job 级别环境变量
    env:
      STRATEGY_NAME: "动量轮动策略"
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.11'

    - name: Install dependencies
      run: |
        pip install -r requirements.txt

    - name: Run signal generation
      # Step 级别环境变量
      env:
        PYTHONUNBUFFERED: "1"
      run: |
        python latest_signal.py --config "$CONFIG_PATH" --strategy "$STRATEGY_NAME"

    - name: Send Feishu webhook alert
      # 使用 Secrets
      env:
        FEISHU_WEBHOOK: ${{ secrets.FEISHU_WEBHOOK_URL }}
        # 使用 Variables
        ALWAYS_NOTIFY: ${{ vars.ALWAYS_NOTIFY }}
      run: |
        # 根据配置决定是否发送
        if [ "$ALWAYS_NOTIFY" = "true" ] || [ "$HAS_REBALANCE" != "0" ]; then
          curl -X POST "$FEISHU_WEBHOOK" \
            -H "Content-Type: application/json" \
            -d '{"msg_type": "text", "content": {"text": "ETF信号"}}'
        fi
```

---

## 环境变量优先级

当同名变量在多个层级定义时，优先级从高到低：

1. **Step 级别** `env:`（最高优先级）
2. **Job 级别** `env:`
3. **Workflow 级别** `env:`
4. **Repository Variables** `vars.XXX`
5. **Repository Secrets** `secrets.XXX`（仅用于敏感信息）

---

## 调试环境变量

在 workflow 中查看所有环境变量：

```yaml
- name: Debug environment variables
  run: |
    echo "=== GitHub 默认变量 ==="
    echo "Repository: $GITHUB_REPOSITORY"
    echo "Run ID: $GITHUB_RUN_ID"
    echo "Ref: $GITHUB_REF"
    
    echo "=== 自定义 Secrets（只显示名称）==="
    echo "FEISHU_WEBHOOK_URL is set: ${{ secrets.FEISHU_WEBHOOK_URL != '' }}"
    
    echo "=== 自定义 Variables ==="
    echo "SIGNAL_STRATEGY: ${{ vars.SIGNAL_STRATEGY }}"
    echo "ALWAYS_NOTIFY: ${{ vars.ALWAYS_NOTIFY }}"
```

---

## 常见问题

### Q: Secrets 和 Variables 有什么区别？

**Secrets**：
- 加密存储，值不可见
- 适合存储敏感信息（Webhook URL、API Key）
- 在 workflow 中通过 `${{ secrets.XXX }}` 引用
- 一旦设置，无法查看原始值，只能删除重建

**Variables**：
- 明文存储，值可见
- 适合存储非敏感配置（策略名称、开关）
- 在 workflow 中通过 `${{ vars.XXX }}` 引用
- 可以随时查看和修改

### Q: 如何在本地测试环境变量？

使用 GitHub CLI：

```bash
# 设置 secret
gh secret set FEISHU_WEBHOOK_URL --body "https://open.feishu.cn/..."

# 设置 variable
gh variable set SIGNAL_STRATEGY --body "动量轮动策略"

# 查看变量列表
gh variable list

# 查看 secret 列表（不显示值）
gh secret list
```

### Q: 环境变量在 shell 脚本中如何使用？

三种方式：

```yaml
- name: Use variables
  env:
    MY_VAR: "hello"
  run: |
    # 方式1：直接引用（推荐）
    echo "$MY_VAR"
    
    # 方式2：使用 ${} 明确边界
    echo "${MY_VAR}_world"
    
    # 方式3：使用 GitHub 表达式
    echo "${{ env.MY_VAR }}"
```

---

## 相关文档

- [GitHub Actions 环境变量官方文档](https://docs.github.com/en/actions/learn-github-actions/variables)
- [GitHub Actions Secrets 管理](https://docs.github.com/en/actions/security-guides/encrypted-secrets)
- [GitHub CLI 使用指南](https://cli.github.com/manual/)
