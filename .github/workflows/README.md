# GitHub Actions Workflow: ETF Daily Signal Alert

## 概述

此 workflow 每天定时运行 `latest_signal.py`，检测 ETF 调仓信号，并通过飞书 webhook 发送告警。

## 触发条件

- **定时触发**: 每天北京时间 15:30 (UTC 07:30)
- **手动触发**: 支持通过 GitHub Actions 页面手动运行

## 配置步骤

### 1. 配置飞书 Webhook

1. 在飞书群中创建机器人，获取 webhook URL
2. 在 GitHub 仓库设置中添加 Secret:
   - 进入 `Settings` → `Secrets and variables` → `Actions`
   - 点击 `New repository secret`
   - Name: `FEISHU_WEBHOOK_URL`
   - Value: 你的飞书 webhook URL

### 2. 确认 workflow 文件

文件已创建: `.github/workflows/etf-signal-alert.yml`

### 3. 测试运行

1. 进入 GitHub 仓库的 `Actions` 页面
2. 选择 `ETF Daily Signal Alert` workflow
3. 点击 `Run workflow` 手动触发测试

## 消息格式

当检测到调仓信号（有买入或卖出）时，飞书会收到如下消息:

```
🚨 ETF调仓信号 - 2026-06-22

⚠️ 检测到调仓信号，请关注！

调仓详情：
  [买入] 创业板ETF (159915)
  [卖出] 科创50ETF易方达 (588080)

当前持仓：
  持有 国债ETF (511010): 58%
  持有 创业板ETF (159915): 22%
  持有 纳指ETF (513100): 22%
```

## 注意事项

- 只有在检测到调仓（有 `[买入]` 或 `[卖出]`）时才会发送消息
- 持仓不变时不会发送消息，避免打扰
- 每次运行日志会保存为 artifact，保留 7 天
- 如果不需要飞书，可以修改 workflow 使用其他 webhook（企业微信、钉钉等）

## 自定义

### 修改定时时间

编辑 `.github/workflows/etf-signal-alert.yml` 中的 cron 表达式:

```yaml
on:
  schedule:
    # 每天北京时间 09:00 运行（UTC 01:00）
    - cron: '0 1 * * *'
```

### 修改告警条件

默认只在有调仓时发送消息。如需每天发送（包括持仓不变），修改:

```yaml
- name: Send Feishu webhook alert
  # 删除或注释掉这行，取消条件限制
  # if: env.HAS_REBALANCE != '0'
```

### 使用其他 Webhook

#### 企业微信

```yaml
- name: Send WeCom webhook alert
  if: env.HAS_REBALANCE != '0'
  env:
    WECOM_WEBHOOK: ${{ secrets.WECOM_WEBHOOK_URL }}
  run: |
    curl -X POST "$WECOM_WEBHOOK" \
      -H "Content-Type: application/json" \
      -d "{
        \"msgtype\": \"text\",
        \"text\": {
          \"content\": \"ETF调仓信号 - ${{ env.SIGNAL_DATE }}\\n\\n$(cat signal_output.txt | head -50)\"
        }
      }"
```

#### 钉钉

```yaml
- name: Send DingTalk webhook alert
  if: env.HAS_REBALANCE != '0'
  env:
    DINGTALK_WEBHOOK: ${{ secrets.DINGTALK_WEBHOOK_URL }}
  run: |
    curl -X POST "$DINGTALK_WEBHOOK" \
      -H "Content-Type: application/json" \
      -d "{
        \"msgtype\": \"text\",
        \"text\": {
          \"content\": \"ETF调仓信号 - ${{ env.SIGNAL_DATE }}\\n\\n$(cat signal_output.txt | head -50)\"
        }
      }"
```
